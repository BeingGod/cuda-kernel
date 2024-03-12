#include "error.cuh"
#include "macro.h"

#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <iostream>
#include <type_traits>

#define WARMUP_ITER 10
#define TEST_ITER 100

using namespace std;

void elementwise_add_kernel_cpu(const float *a, const float *b, float *c,
                                size_t len) {
  for (int i = 0; i < len; ++i) {
    c[i] = a[i] + b[i];
  }
}

__global__ void elementwise_add_kernel_gpu_naive(const float *a, const float *b,
                                                 float *c, size_t len) {
  CUDA_KERNEL_LOOP(i, len) { c[i] = a[i] + b[i]; }
}

__global__ void elementwise_add_kernel_gpu_opt1(const float *a, const float *b,
                                                float *c, size_t len) {
  auto idx = blockIdx.x * blockDim.x + threadIdx.x;
  for (auto i = idx; i < len / 4; i += blockDim.x * gridDim.x) {
    float4 a4 = reinterpret_cast<const float4 *>(a)[i];
    float4 b4 = reinterpret_cast<const float4 *>(b)[i];
    float4 c4;
    c4.x = a4.x + b4.x;
    c4.y = a4.y + b4.y;
    c4.z = a4.z + b4.z;
    c4.w = a4.w + b4.w;
    reinterpret_cast<float4 *>(c)[i] = c4;
  }

  // in only one thread, process final elements (if there are any)
  auto remainder = len % 4;
  if (idx == len / 4 && remainder != 0) {
    while (remainder) {
      auto idx = len - remainder--;
      c[idx] = a[idx] + b[idx];
    }
  }
}

__global__ void elementwise_add_kernel_gpu_opt2(const float *a, const float *b,
                                                float *c, size_t len) {
#pragma unroll
  for (auto i = blockIdx.x * blockDim.x + threadIdx.x; i < len;
       i += blockDim.x * gridDim.x) {
    c[i] = a[i] + b[i];
  }
}

__global__ void elementwise_add_kernel_gpu_opt3(const float *a, const float *b,
                                                float *c, size_t len) {
  auto idx = blockIdx.x * blockDim.x + threadIdx.x;
#pragma unroll
  for (auto i = idx; i < len / 4; i += blockDim.x * gridDim.x) {
    float4 a4 = reinterpret_cast<const float4 *>(a)[i];
    float4 b4 = reinterpret_cast<const float4 *>(b)[i];
    float4 c4;
    c4.x = a4.x + b4.x;
    c4.y = a4.y + b4.y;
    c4.z = a4.z + b4.z;
    c4.w = a4.w + b4.w;
    reinterpret_cast<float4 *>(c)[i] = c4;
  }

  // in only one thread, process final elements (if there are any)
  auto remainder = len % 4;
  if (idx == len / 4 && remainder != 0) {
    while (remainder) {
      auto idx = len - remainder--;
      c[idx] = a[idx] + b[idx];
    }
  }
}

void elementwise_add_gpu_naive(const float *a, const float *b, float *c,
                               size_t len) {
  elementwise_add_kernel_gpu_naive<<<BLOCKS_PER_GRID(len), THREADS_PER_BLOCK>>>(
      a, b, c, len);
  CHECK_CUDA_ERROR(cudaGetLastError());
}

void elementwise_add_gpu_1(const float *a, const float *b, float *c,
                           size_t len) {
  elementwise_add_kernel_gpu_opt1<<<BLOCKS_PER_GRID(len) / 4,
                                    THREADS_PER_BLOCK>>>(a, b, c, len);
  CHECK_CUDA_ERROR(cudaGetLastError());
}

void elementwise_add_gpu_2(const float *a, const float *b, float *c,
                           size_t len) {
  elementwise_add_kernel_gpu_opt2<<<BLOCKS_PER_GRID(len), THREADS_PER_BLOCK>>>(
      a, b, c, len);
  CHECK_CUDA_ERROR(cudaGetLastError());
}

void elementwise_add_gpu_3(const float *a, const float *b, float *c,
                           size_t len) {
  elementwise_add_kernel_gpu_opt3<<<BLOCKS_PER_GRID(len) / 4,
                                    THREADS_PER_BLOCK>>>(a, b, c, len);
  CHECK_CUDA_ERROR(cudaGetLastError());
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
  float *b = (float *)malloc(size);
  float *cpu_c = (float *)malloc(size);
  memset(cpu_c, 0, sizeof(float) * len);
  float *gpu_c = (float *)malloc(size);
  memset(gpu_c, 0, sizeof(float) * len);

  init_random(a, len);
  init_random(b, len);

  float *d_a, *d_b, *d_c;
  CHECK_CUDA_ERROR(cudaMalloc(&d_a, size));
  CHECK_CUDA_ERROR(cudaMalloc(&d_b, size));
  CHECK_CUDA_ERROR(cudaMalloc(&d_c, size));

  CHECK_CUDA_ERROR(cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice));
  CHECK_CUDA_ERROR(cudaMemcpy(d_b, b, size, cudaMemcpyHostToDevice));

  for (int i = 0; i < WARMUP_ITER; ++i) {
    elementwise_add_kernel_gpu_naive<<<BLOCKS_PER_GRID(len),
                                       THREADS_PER_BLOCK>>>(d_a, d_b, d_c, len);
  }

  elementwise_add_kernel_cpu(a, b, cpu_c, len);

  BENCHMARK(elementwise_add_gpu_naive, TEST_ITER, d_a, d_b, d_c, len);
  CHECK_CUDA_ERROR(cudaMemcpy(gpu_c, d_c, size, cudaMemcpyDeviceToHost));
  compare(gpu_c, cpu_c, len);

  BENCHMARK(elementwise_add_gpu_1, TEST_ITER, d_a, d_b, d_c, len);
  CHECK_CUDA_ERROR(cudaMemcpy(gpu_c, d_c, size, cudaMemcpyDeviceToHost));
  compare(gpu_c, cpu_c, len);

  BENCHMARK(elementwise_add_gpu_2, TEST_ITER, d_a, d_b, d_c, len);
  CHECK_CUDA_ERROR(cudaMemcpy(gpu_c, d_c, size, cudaMemcpyDeviceToHost));
  compare(gpu_c, cpu_c, len);

  BENCHMARK(elementwise_add_gpu_3, TEST_ITER, d_a, d_b, d_c, len);
  CHECK_CUDA_ERROR(cudaMemcpy(gpu_c, d_c, size, cudaMemcpyDeviceToHost));
  compare(gpu_c, cpu_c, len);

  CHECK_CUDA_ERROR(cudaFree(d_a));
  CHECK_CUDA_ERROR(cudaFree(d_b));
  CHECK_CUDA_ERROR(cudaFree(d_c));

  return 0;
}