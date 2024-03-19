#include "error.cuh"
#include "macro.h"

#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <iostream>
#include <math.h>
#include <type_traits>

#define WARMUP_ITER 10
#define TEST_ITER 100

using namespace std;

void softmax_kernel_cpu(const float *x, float *y, const int n) {
  double sum = 0;
  for (int i = 0; i < n; ++i) {
    sum += expf(x[i]);
  }

  for (int i = 0; i < n; ++i) {
    y[i] = expf(x[i]) / static_cast<float>(sum);
  }
}

void safe_softmax_kernel_cpu(const float *x, float *y, const int n) {
  float max_x = -INFINITY;
  for (int i = 0; i < n; ++i) {
    max_x = max(x[i], max_x);
  }

  double sum = 0;
  for (int i = 0; i < n; ++i) {
    sum += expf(x[i] - max_x);
  }

  for (int i = 0; i < n; ++i) {
    y[i] = expf(x[i] - max_x) / static_cast<float>(sum);
  }
}

void online_softmax_kernel_cpu(const float *x, float *y, const int n) {
  double sum = 0;
  float max_x = -INFINITY;
  for (int i = 0; i < n; ++i) {
    float new_max_x = max(max_x, x[i]);
    sum = sum * expf(max_x - new_max_x) + expf(x[i] - new_max_x);
    max_x = new_max_x;
  }

  for (int i = 0; i < n; ++i) {
    y[i] = expf(x[i] - max_x) / static_cast<float>(sum);
  }
}

template <typename T> __inline__ __device__ T warp_reduce_max(T val) {
#pragma unroll
  for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
    val = fmaxf(val, __shfl_down_sync(FULL_MASK, val, offset));
  }

  return val;
}

template <typename T> __inline__ __device__ T warp_reduce_sum(T val) {
#pragma unroll
  for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
    val += __shfl_down_sync(FULL_MASK, val, offset);
  }

  return val;
}

template <typename T> __inline__ __device__ T block_reduce_max(T val) {
  __shared__ T s_tmp[32];
  auto wid = threadIdx.x / warpSize;
  auto lane = threadIdx.x % warpSize;

  val = warp_reduce_max(val);

  if (lane == 0) {
    s_tmp[wid] = val;
  }

  __syncthreads();

  val = threadIdx.x < blockDim.x / warpSize ? s_tmp[lane] : 0;

  if (wid == 0) {
    val = warp_reduce_max(val);
  }

  return val;
}

template <typename T> __inline__ __device__ T block_reduce_sum(T val) {
  __shared__ T s_tmp[32];
  auto wid = threadIdx.x / warpSize;
  auto lane = threadIdx.x % warpSize;

  val = warp_reduce_sum(val);

  if (lane == 0) {
    s_tmp[wid] = val;
  }

  __syncthreads();

  val = threadIdx.x < blockDim.x / warpSize ? s_tmp[lane] : 0;

  if (wid == 0) {
    val = warp_reduce_sum(val);
  }

  return val;
}

__global__ void softmax_kernel_gpu_1(const float *__restrict__ x,
                                     float *__restrict__ y, float *global_sum,
                                     const int n) {
  float sum = 0;
  for (auto i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
       i += gridDim.x * blockDim.x) {
    sum += expf(x[i]);
  }

  sum = block_reduce_sum<float>(sum);

  if (threadIdx.x == 0) {
    atomicAdd(global_sum, sum);
  }

  __threadfence();

  for (auto i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
       i += gridDim.x * blockDim.x) {
    y[i] = expf(x[i]) / global_sum[0];
  }
}

__global__ void softmax_kernel_gpu_2(const float *__restrict__ x,
                                     float *__restrict__ y, float *global_sum,
                                     const int n) {
  float sum = 0;
  auto idx = blockIdx.x * blockDim.x + threadIdx.x;
  for (auto i = idx; i < n / 4; i += gridDim.x * blockDim.x) {
    float4 val = CONST_FLOAT4(x)[i];

    sum += expf(val.x);
    sum += expf(val.y);
    sum += expf(val.z);
    sum += expf(val.w);
  }
  auto i = idx + n / 4 * 4;
  if (i < n) {
    sum += expf(x[i]);
  }

  sum = block_reduce_sum<float>(sum);

  if (threadIdx.x == 0) {
    atomicAdd(global_sum, sum);
  }

  __threadfence();

  const float r_globl_sum = global_sum[0];
  for (auto i = idx; i < n / 4; i += gridDim.x * blockDim.x) {
    float4 val = CONST_FLOAT4(x)[i];
    val.x = expf(val.x) / r_globl_sum;
    val.y = expf(val.y) / r_globl_sum;
    val.z = expf(val.z) / r_globl_sum;
    val.w = expf(val.w) / r_globl_sum;
    FLOAT4(y)[i] = val;
  }
  i = idx + n / 4 * 4;
  if (i < n) {
    y[i] = expf(x[i]) / r_globl_sum;
  }
}

__global__ void softmax_kernel_gpu_3(const float *__restrict__ x,
                                     float *__restrict__ y, float *global_sum,
                                     const int n,
                                     const float *__restrict__ max_x) {
  float s_max_x = max_x[0];

  float sum = 0;
  auto idx = blockIdx.x * blockDim.x + threadIdx.x;
  for (auto i = idx; i < n / 4; i += gridDim.x * blockDim.x) {
    float4 val = CONST_FLOAT4(x)[i];

    sum += expf(val.x - s_max_x);
    sum += expf(val.y - s_max_x);
    sum += expf(val.z - s_max_x);
    sum += expf(val.w - s_max_x);
  }
  auto i = idx + n / 4 * 4;
  if (i < n) {
    sum += expf(x[i] - s_max_x);
  }

  sum = block_reduce_sum<float>(sum);

  if (threadIdx.x == 0) {
    atomicAdd(global_sum, sum);
  }

  __threadfence();

  const float r_globl_sum = global_sum[0];
  for (auto i = idx; i < n / 4; i += gridDim.x * blockDim.x) {
    float4 val = CONST_FLOAT4(x)[i];
    val.x = expf(val.x - s_max_x) / r_globl_sum;
    val.y = expf(val.y - s_max_x) / r_globl_sum;
    val.z = expf(val.z - s_max_x) / r_globl_sum;
    val.w = expf(val.w - s_max_x) / r_globl_sum;
    FLOAT4(y)[i] = val;
  }
  i = idx + n / 4 * 4;
  if (i < n) {
    y[i] = expf(x[i] - s_max_x) / r_globl_sum;
  }
}

__global__ void reduce_max_kernel_gpu(const float *__restrict__ x,
                                      float *__restrict__ max_x, const int n) {
  float val = -INFINITY;
  auto idx = blockIdx.x * blockDim.x + threadIdx.x;
  for (auto i = idx; i < n / 4; i += gridDim.x * blockDim.x) {
    float4 v_x = CONST_FLOAT4(x)[i];

    val = fmaxf(val, v_x.x);
    val = fmaxf(val, v_x.y);
    val = fmaxf(val, v_x.z);
    val = fmaxf(val, v_x.w);
  }
  auto i = idx + n / 4 * 4;
  if (i < n) {
    val = fmaxf(val, x[i]);
  }

  val = block_reduce_max<float>(val);

  if (threadIdx.x == 0) {
    max_x[blockIdx.x] = val;
  }
}

void softmax_gpu_1(const float *x, float *y, const int n) {
  float *d_global_sum;
  CHECK_CUDA_ERROR(cudaMalloc(&d_global_sum, sizeof(float)));
  CHECK_CUDA_ERROR(cudaMemset(d_global_sum, 0, sizeof(float)));
  softmax_kernel_gpu_1<<<BLOCKS_PER_GRID(n), THREADS_PER_BLOCK>>>(
      x, y, d_global_sum, n);
  cudaStreamSynchronize(0);
  CHECK_CUDA_ERROR(cudaGetLastError());
}

void softmax_gpu_2(const float *x, float *y, const int n) {
  float *d_global_sum;
  CHECK_CUDA_ERROR(cudaMalloc(&d_global_sum, sizeof(float)));
  CHECK_CUDA_ERROR(cudaMemset(d_global_sum, 0, sizeof(float)));
  softmax_kernel_gpu_2<<<max(BLOCKS_PER_GRID(n) / 4, 1), THREADS_PER_BLOCK>>>(
      x, y, d_global_sum, n);
  cudaStreamSynchronize(0);
  CHECK_CUDA_ERROR(cudaGetLastError());
  CHECK_CUDA_ERROR(cudaFree(d_global_sum));
}

void softmax_gpu_3(const float *x, float *y, const int n) {
  // safe softmax
  float *d_tmp, *d_max_x;
  CHECK_CUDA_ERROR(cudaMalloc(&d_tmp, sizeof(float) * THREADS_PER_BLOCK));
  CHECK_CUDA_ERROR(cudaMalloc(&d_max_x, sizeof(float)));

  auto blocks_per_grid = max(BLOCKS_PER_GRID(n) / 4, 1);
  reduce_max_kernel_gpu<<<blocks_per_grid, THREADS_PER_BLOCK>>>(x, d_tmp, n);

  auto threads_per_block = max(BLOCKS_PER_GRID(n) / 4, 1024);
  reduce_max_kernel_gpu<<<1, threads_per_block>>>(d_tmp, d_max_x,
                                                  blocks_per_grid);
  float *d_global_sum;
  CHECK_CUDA_ERROR(cudaMalloc(&d_global_sum, sizeof(float)));
  CHECK_CUDA_ERROR(cudaMemset(d_global_sum, 0, sizeof(float)));
  softmax_kernel_gpu_3<<<max(BLOCKS_PER_GRID(n) / 4, 1), THREADS_PER_BLOCK>>>(
      x, y, d_global_sum, n, d_max_x);
  cudaStreamSynchronize(0);
  CHECK_CUDA_ERROR(cudaGetLastError());

  CHECK_CUDA_ERROR(cudaFree(d_global_sum));
  CHECK_CUDA_ERROR(cudaFree(d_tmp));
  CHECK_CUDA_ERROR(cudaFree(d_max_x));
}

void init_random(float *data, size_t len) {
  for (size_t i = 0; i < len; ++i) {
    data[i] = static_cast<float>(std::rand()) / static_cast<float>(RAND_MAX);
  }
}

void compare(float *res1, float *res2, size_t len) {
  for (size_t i = 0; i < len; ++i) {
    double diff = std::abs(res1[i] - res2[i]);
    if (std::is_same<float, double>::value) {
      if (diff > 1e-7) {
        fprintf(stderr, "check failed ! index: %lu, res1: %.13f res2: %.13f\n",
                i, res1[i], res2[i]);
        return;
      }
    } else {
      if (diff > 1e-4) {
        fprintf(stderr, "check failed ! index: %lu, res1: %.7f res2: %.7f\n", i,
                res1[i], res2[i]);
        return;
      }
    }
  }
  printf("check pass !\n");
}

int main(int argc, char *argv[]) {
  if (argc < 2) {
    std::cout << "[Useage] " << argv[0] << " "
              << "<data len>" << std::endl;
    std::exit(1);
  }
  int num_sm;
  cudaDeviceGetAttribute(&num_sm, cudaDevAttrMultiProcessorCount, 0);

  std::srand(std::time(nullptr));

  size_t len = atol(argv[1]);
  size_t size = len * sizeof(float);

  float *a = (float *)malloc(size);
  float *gpu_b = (float *)malloc(size);
  float *cpu_b = (float *)malloc(size);

  init_random(a, len);

  // softmax_kernel_cpu(a, cpu_b, len);
  // safe_softmax_kernel_cpu(a, cpu_b, len);
  online_softmax_kernel_cpu(a, cpu_b, len);

  float *d_a, *d_b;
  CHECK_CUDA_ERROR(cudaMalloc(&d_a, size));
  CHECK_CUDA_ERROR(cudaMalloc(&d_b, size));

  CHECK_CUDA_ERROR(cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice));

  for (int i = 0; i < WARMUP_ITER; ++i) {
    float *d_global_sum;
    CHECK_CUDA_ERROR(cudaMalloc(&d_global_sum, sizeof(double)));
    softmax_kernel_gpu_1<<<BLOCKS_PER_GRID(len), THREADS_PER_BLOCK>>>(
        d_a, d_b, d_global_sum, len);
  }

  BENCHMARK(softmax_gpu_1, TEST_ITER, d_a, d_b, len);
  CHECK_CUDA_ERROR(cudaMemcpy(gpu_b, d_b, size, cudaMemcpyDeviceToHost));
  compare(cpu_b, gpu_b, len);

  BENCHMARK(softmax_gpu_2, TEST_ITER, d_a, d_b, len);
  CHECK_CUDA_ERROR(cudaMemcpy(gpu_b, d_b, size, cudaMemcpyDeviceToHost));
  compare(cpu_b, gpu_b, len);

  BENCHMARK(softmax_gpu_3, TEST_ITER, d_a, d_b, len);
  CHECK_CUDA_ERROR(cudaMemcpy(gpu_b, d_b, size, cudaMemcpyDeviceToHost));
  compare(cpu_b, gpu_b, len);

  CHECK_CUDA_ERROR(cudaFree(d_a));
  CHECK_CUDA_ERROR(cudaFree(d_b));

  return 0;
}