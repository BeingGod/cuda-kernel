#include "error.cuh"
#include "macro.h"

#include <cassert>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <iostream>
#include <type_traits>

#define DATATYPE float
#define WARMUP_ITER 3
#define TEST_ITER 100

using namespace std;

void reduce_sum_cpu(const DATATYPE *a, DATATYPE *c, size_t len) {
  double temp = 0;
  for (int i = 0; i < len; ++i) {
    temp += a[i];
  }
  *c = static_cast<DATATYPE>(temp);
}

__global__ void reduce_sum_gpu_1(const DATATYPE *a, DATATYPE *b, size_t len) {
  __shared__ DATATYPE s_tmp[THREADS_PER_BLOCK];
  auto tid = threadIdx.x;
  auto idx = blockDim.x * blockIdx.x + threadIdx.x;
  double temp = 0;
  for (int i = idx; i < len; i += blockDim.x * gridDim.x) {
    temp += a[i];
  }
  s_tmp[tid] = temp;
  __syncthreads();

  int i = 2, j = 1;
  while (i <= THREADS_PER_BLOCK) {
    // tree reduction
    if ((tid % i) == 0) {
      s_tmp[tid] += s_tmp[tid + j];
    }
    __syncthreads();
    i *= 2;
    j *= 2;
  }
  if (tid == 0) {
    b[tid] = s_tmp[0];
  }
}

__global__ void reduce_sum_gpu_2(const DATATYPE *a, DATATYPE *b, size_t len) {
  __shared__ DATATYPE s_tmp[THREADS_PER_BLOCK];
  auto tid = threadIdx.x;
  auto idx = blockDim.x * blockIdx.x + threadIdx.x;
  double temp = 0;
  for (int i = idx; i < len; i += blockDim.x * gridDim.x) {
    temp += a[i];
  }
  s_tmp[tid] = temp;
  __syncthreads();

  int offset = THREADS_PER_BLOCK / 2;
  for (; offset > 0; offset >>= 1) {
    if (tid < offset) {
      s_tmp[tid] += s_tmp[tid + offset];
    }
    __syncthreads();
  }
  if (tid == 0) {
    b[tid] = s_tmp[0];
  }
}

__global__ void reduce_sum_gpu_3(const DATATYPE *a, DATATYPE *b, size_t len) {
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    b[0] = 0;
  }
  __shared__ DATATYPE s_tmp[THREADS_PER_BLOCK];
  auto tid = threadIdx.x;
  auto idx = blockDim.x * blockIdx.x + threadIdx.x;
  double temp = 0;
  for (int i = idx; i < len; i += blockDim.x * gridDim.x) {
    temp += a[i];
  }
  s_tmp[tid] = temp;
  __syncthreads();

  int offset = THREADS_PER_BLOCK / 2;
  for (; offset > 0; offset >>= 1) {
    if (tid < offset) {
      s_tmp[tid] += s_tmp[tid + offset];
    }
    __syncthreads();
  }
  if (tid == 0) {
    atomicAdd(b, s_tmp[0]);
  }
}

__global__ void reduce_sum_gpu_4(const DATATYPE *a, DATATYPE *b, size_t len) {
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    b[0] = 0;
  }
  __shared__ DATATYPE s_tmp[THREADS_PER_BLOCK];
  auto tid = threadIdx.x;
  auto idx = (blockDim.x * 2) * blockIdx.x + threadIdx.x;
  double temp = 0;
  for (int i = idx; i < len; i += blockDim.x * 2 * gridDim.x) {
    temp += a[i] + a[i + blockDim.x];
  }
  s_tmp[tid] = temp;
  __syncthreads();

  int offset = THREADS_PER_BLOCK / 2;
  for (; offset > 0; offset >>= 1) {
    if (tid < offset) {
      s_tmp[tid] += s_tmp[tid + offset];
    }
    __syncthreads();
  }
  if (tid == 0) {
    atomicAdd(b, s_tmp[0]);
  }
}

__device__ void warp_reduce(volatile DATATYPE *s_data, size_t tid) {
  s_data[tid] += s_data[tid + 32];
  s_data[tid] += s_data[tid + 16];
  s_data[tid] += s_data[tid + 8];
  s_data[tid] += s_data[tid + 4];
  s_data[tid] += s_data[tid + 2];
  s_data[tid] += s_data[tid + 1];
}

__global__ void reduce_sum_gpu_5(const DATATYPE *a, DATATYPE *b, size_t len) {
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    b[0] = 0;
  }
  __shared__ DATATYPE s_tmp[THREADS_PER_BLOCK];
  auto tid = threadIdx.x;
  auto idx = (blockDim.x * 2) * blockIdx.x + threadIdx.x;
  double temp = 0;
  for (int i = idx; i < len; i += blockDim.x * 2 * gridDim.x) {
    temp += a[i] + a[i + blockDim.x];
  }
  s_tmp[tid] = temp;
  __syncthreads();

  int offset = THREADS_PER_BLOCK / 2;
  for (; offset > 32; offset >>= 1) {
    if (tid < offset) {
      s_tmp[tid] += s_tmp[tid + offset];
    }
    __syncthreads();
  }

  if (tid < 32) {
    warp_reduce(s_tmp, tid);
  }

  if (tid == 0) {
    atomicAdd(b, s_tmp[0]);
  }
}

__global__ void reduce_sum_gpu_5_1(const DATATYPE *a, DATATYPE *b, size_t len) {
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    b[0] = 0;
  }
  __shared__ DATATYPE s_tmp[THREADS_PER_BLOCK];
  auto tid = threadIdx.x;
  auto idx = blockDim.x * blockIdx.x + threadIdx.x;
  double temp = 0;
  float4 val;
  for (int i = idx; i < len / 4; i += blockDim.x * gridDim.x) {
    val = CONST_FLOAT4(a)[i];
    temp += ((val.x + val.y) + (val.z + val.w));
  }
  int i = idx + len / 4 * 4;
  if (i < len) {
    temp += a[i];
  }

  s_tmp[tid] = temp;
  __syncthreads();

  int offset = THREADS_PER_BLOCK / 2;
  for (; offset > 32; offset >>= 1) {
    if (tid < offset) {
      s_tmp[tid] += s_tmp[tid + offset];
    }
    __syncthreads();
  }

  if (tid < 32) {
    warp_reduce(s_tmp, tid);
  }

  if (tid == 0) {
    atomicAdd(b, s_tmp[0]);
  }
}

__inline__ __device__ DATATYPE warp_reduce_sum(DATATYPE val) {
#pragma unroll
  for (int offset = warpSize / 2; offset > 0; offset = offset >> 1) {
    val += __shfl_down_sync(FULL_MASK, val, offset);
  }

  return val;
}

__inline__ __device__ DATATYPE block_reduce_sum(DATATYPE val) {
  __shared__ DATATYPE s_tmp[32];

  int wid = threadIdx.x / warpSize;
  int lane = threadIdx.x % warpSize;

  val = warp_reduce_sum(val);

  if (lane == 0) {
    s_tmp[wid] = val;
  }

  __syncthreads();

  val = (threadIdx.x < blockDim.x / warpSize) ? s_tmp[lane] : 0;

  if (wid == 0) {
    val = warp_reduce_sum(val);
  }

  return val;
}

__global__ void reduce_sum_gpu_6(const DATATYPE *a, DATATYPE *b, size_t len) {
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    b[0] = 0;
  }
  auto tid = threadIdx.x;
  auto idx = (blockDim.x * 2) * blockIdx.x + threadIdx.x;
  double sum = 0;
  for (int i = idx; i < len; i += blockDim.x * 2 * gridDim.x) {
    sum += a[i] + a[i + blockDim.x];
  }

  sum = block_reduce_sum(sum);

  if (tid == 0) {
    atomicAdd(b, sum);
  }
}

__global__ void reduce_sum_gpu_6_1(const DATATYPE *a, DATATYPE *b, size_t len) {
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    b[0] = 0;
  }
  auto tid = threadIdx.x;
  auto idx = blockDim.x * blockIdx.x + threadIdx.x;
  double sum = 0;
  float4 val;
  for (int i = idx; i < len / 4; i += blockDim.x * gridDim.x) {
    val = CONST_FLOAT4(a)[i];
    sum += ((val.x + val.y) + (val.z + val.w));
  }

  sum = block_reduce_sum(sum);

  if (tid == 0) {
    atomicAdd(b, sum);
  }
}

void init_random(DATATYPE *data, size_t len) {
  for (size_t i = 0; i < len; ++i) {
    // data[i] =
    //     static_cast<DATATYPE>(std::rand()) / static_cast<DATATYPE>(RAND_MAX);
    data[i] = 1;
  }
}

void compare(DATATYPE res1, DATATYPE res2) {
  printf("%.7f, %.7f\n", res1, res2);

  double diff = std::abs(res1 - res2);
  if (std::is_same<DATATYPE, double>::value) {
    assert(diff < 1e-13);
  } else {
    assert(diff < 1e-6);
  }
}

void reduce_sum_1(const DATATYPE *d_a, DATATYPE *d_b, DATATYPE *d_c,
                  size_t len) {
  reduce_sum_gpu_1<<<BLOCKS_PER_GRID(len), THREADS_PER_BLOCK>>>(d_a, d_b, len);
  CHECK_CUDA_ERROR(cudaGetLastError());
  reduce_sum_gpu_1<<<1, THREADS_PER_BLOCK>>>(d_b, d_c, THREADS_PER_BLOCK);
  CHECK_CUDA_ERROR(cudaGetLastError());
}

void reduce_sum_2(const DATATYPE *d_a, DATATYPE *d_b, DATATYPE *d_c,
                  size_t len) {
  reduce_sum_gpu_2<<<BLOCKS_PER_GRID(len), THREADS_PER_BLOCK>>>(d_a, d_b, len);
  CHECK_CUDA_ERROR(cudaGetLastError());
  reduce_sum_gpu_2<<<1, THREADS_PER_BLOCK>>>(d_b, d_c, THREADS_PER_BLOCK);
  CHECK_CUDA_ERROR(cudaGetLastError());
}

void reduce_sum_3(const DATATYPE *d_a, DATATYPE *d_b, DATATYPE *d_c,
                  size_t len) {
  reduce_sum_gpu_3<<<BLOCKS_PER_GRID(len), THREADS_PER_BLOCK>>>(d_a, d_c, len);
  CHECK_CUDA_ERROR(cudaGetLastError());
}

void reduce_sum_4(const DATATYPE *d_a, DATATYPE *d_b, DATATYPE *d_c,
                  size_t len) {
  reduce_sum_gpu_4<<<BLOCKS_PER_GRID(len), THREADS_PER_BLOCK>>>(d_a, d_c, len);
  CHECK_CUDA_ERROR(cudaGetLastError());
}

void reduce_sum_5(const DATATYPE *d_a, DATATYPE *d_b, DATATYPE *d_c,
                  size_t len) {
  reduce_sum_gpu_5<<<BLOCKS_PER_GRID(len), THREADS_PER_BLOCK>>>(d_a, d_c, len);
  CHECK_CUDA_ERROR(cudaGetLastError());
}

void reduce_sum_5_1(const DATATYPE *d_a, DATATYPE *d_b, DATATYPE *d_c,
                    size_t len) {
  reduce_sum_gpu_5_1<<<
      MIN((len / 4 + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK, 2048),
      THREADS_PER_BLOCK>>>(d_a, d_c, len);
  CHECK_CUDA_ERROR(cudaGetLastError());
}

void reduce_sum_6(const DATATYPE *d_a, DATATYPE *d_b, DATATYPE *d_c,
                  size_t len) {
  reduce_sum_gpu_6<<<BLOCKS_PER_GRID(len), THREADS_PER_BLOCK>>>(d_a, d_c, len);
  CHECK_CUDA_ERROR(cudaGetLastError());
}

void reduce_sum_6_1(const DATATYPE *d_a, DATATYPE *d_b, DATATYPE *d_c,
                    size_t len) {
  reduce_sum_gpu_6_1<<<
      MIN((len / 4 + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK, 2048),
      THREADS_PER_BLOCK>>>(d_a, d_c, len);
  CHECK_CUDA_ERROR(cudaGetLastError());
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
  size_t size = len * sizeof(DATATYPE);

  DATATYPE *a = (DATATYPE *)malloc(size);
  DATATYPE *cpu_c = (DATATYPE *)malloc(1 * sizeof(DATATYPE));
  DATATYPE *gpu_c = (DATATYPE *)malloc(1 * sizeof(DATATYPE));

  init_random(a, len);

  DATATYPE *d_a, *d_b, *d_c;
  CHECK_CUDA_ERROR(cudaMalloc(&d_a, size));
  CHECK_CUDA_ERROR(cudaMalloc(&d_b, sizeof(DATATYPE) * THREADS_PER_BLOCK));
  CHECK_CUDA_ERROR(cudaMalloc(&d_c, sizeof(DATATYPE) * 1));

  CHECK_CUDA_ERROR(cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice));

  for (int i = 0; i < WARMUP_ITER; ++i) {
    reduce_sum_1(d_a, d_b, d_c, len);
  }

  // BENCHMARK(reduce_sum_1, TEST_ITER, d_a, d_b, d_c, len);
  // BENCHMARK(reduce_sum_2, TEST_ITER, d_a, d_b, d_c, len);
  // BENCHMARK(reduce_sum_3, TEST_ITER, d_a, d_b, d_c, len);
  // BENCHMARK(reduce_sum_4, TEST_ITER, d_a, d_b, d_c, len);
  // BENCHMARK(reduce_sum_5, TEST_ITER, d_a, d_b, d_c, len);
  // BENCHMARK(reduce_sum_5_1, TEST_ITER, d_a, d_b, d_c, len);
  BENCHMARK(reduce_sum_6, TEST_ITER, d_a, d_b, d_c, len);
  // BENCHMARK(reduce_sum_6_1, TEST_ITER, d_a, d_b, d_c, len);

  CHECK_CUDA_ERROR(
      cudaMemcpy(gpu_c, d_c, sizeof(DATATYPE) * 1, cudaMemcpyDeviceToHost));

  CHECK_CUDA_ERROR(cudaFree(d_a));
  CHECK_CUDA_ERROR(cudaFree(d_b));
  CHECK_CUDA_ERROR(cudaFree(d_c));

  reduce_sum_cpu(a, cpu_c, len);

  compare(*gpu_c, *cpu_c);

  return 0;
}