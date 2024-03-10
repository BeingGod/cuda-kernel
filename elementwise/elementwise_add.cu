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

#define DATATYPE float
#define WARMUP_ITER 10
#define TEST_ITER 100

using namespace std;

void elementwise_add_cpu(const DATATYPE *a, const DATATYPE *b, DATATYPE *c,
                         size_t len) {
  for (int i = 0; i < len; ++i) {
    c[i] = a[i] + b[i];
  }
}

__global__ void elementwise_add_gpu_naive(const DATATYPE *a, const DATATYPE *b,
                                          DATATYPE *c, size_t len) {
  CUDA_KERNEL_LOOP(i, len) { c[i] = a[i] + b[i]; }
}

__global__ void elementwise_add_gpu_opt1(const DATATYPE *a, const DATATYPE *b,
                                         DATATYPE *c, size_t len) {
  int idx = (blockIdx.x * blockDim.x + threadIdx.x) * 4;
  for (int i = idx; i < len / 4; i += blockDim.x * gridDim.x) {
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
  int remainder = len % 4;
  if (idx == len / 4 && remainder != 0) {
    while (remainder) {
      int idx = len - remainder--;
      c[idx] = a[idx] + b[idx];
    }
  }
}

__global__ void elementwise_add_gpu_opt2(const DATATYPE *a, const DATATYPE *b,
                                         DATATYPE *c, size_t len) {
#pragma unroll
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < len;
       i += blockDim.x * gridDim.x) {
    c[i] = a[i] + b[i];
  }
}

__global__ void elementwise_add_gpu_opt3(const DATATYPE *a, const DATATYPE *b,
                                         DATATYPE *c, size_t len) {
  int idx = (blockIdx.x * blockDim.x + threadIdx.x) * 4;
#pragma unroll
  for (int i = idx; i < len / 4; i += blockDim.x * gridDim.x) {
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
  int remainder = len % 4;
  if (idx == len / 4 && remainder != 0) {
    while (remainder) {
      int idx = len - remainder--;
      c[idx] = a[idx] + b[idx];
    }
  }
}

void init_random(DATATYPE *data, size_t len) {
  for (size_t i = 0; i < len; ++i) {
    data[i] =
        static_cast<DATATYPE>(std::rand()) / static_cast<DATATYPE>(RAND_MAX);
  }
}

void compare(DATATYPE *res1, DATATYPE *res2, size_t len) {
  for (size_t i = 0; i < len; ++i) {
    double diff = std::abs(res1[i] - res2[i]);
    if (std::is_same<DATATYPE, double>::value) {
      assert(diff < 1e-13);
    } else {
      assert(diff < 1e-6);
    }
  }
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
  DATATYPE *b = (DATATYPE *)malloc(size);
  DATATYPE *cpu_c = (DATATYPE *)malloc(size);
  memset(cpu_c, 0, sizeof(DATATYPE) * len);
  DATATYPE *gpu_c = (DATATYPE *)malloc(size);
  memset(gpu_c, 0, sizeof(DATATYPE) * len);

  init_random(a, len);
  init_random(b, len);

  DATATYPE *d_a, *d_b, *d_c;
  CHECK_CUDA_ERROR(cudaMalloc(&d_a, size));
  CHECK_CUDA_ERROR(cudaMalloc(&d_b, size));
  CHECK_CUDA_ERROR(cudaMalloc(&d_c, size));

  CHECK_CUDA_ERROR(cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice));
  CHECK_CUDA_ERROR(cudaMemcpy(d_b, b, size, cudaMemcpyHostToDevice));

  for (int i = 0; i < WARMUP_ITER; ++i) {
    elementwise_add_gpu_naive<<<BLOCKS_PER_GRID(len), THREADS_PER_BLOCK>>>(
        d_a, d_b, d_c, len);
  }

  for (int i = 0; i < TEST_ITER; ++i) {
    elementwise_add_gpu_naive<<<BLOCKS_PER_GRID(len), THREADS_PER_BLOCK>>>(
        d_a, d_b, d_c, len);
  }

  for (int i = 0; i < TEST_ITER; ++i) {
    elementwise_add_gpu_opt1<<<BLOCKS_PER_GRID(len) / 4, THREADS_PER_BLOCK>>>(
        d_a, d_b, d_c, len);
  }

  for (int i = 0; i < TEST_ITER; ++i) {
    elementwise_add_gpu_opt2<<<BLOCKS_PER_GRID(len), THREADS_PER_BLOCK>>>(
        d_a, d_b, d_c, len);
  }

  for (int i = 0; i < TEST_ITER; ++i) {
    elementwise_add_gpu_opt3<<<BLOCKS_PER_GRID(len) / 4, THREADS_PER_BLOCK>>>(
        d_a, d_b, d_c, len);
  }

  CHECK_CUDA_ERROR(cudaGetLastError());

  CHECK_CUDA_ERROR(cudaMemcpy(gpu_c, d_c, size, cudaMemcpyDeviceToHost));

  CHECK_CUDA_ERROR(cudaFree(d_a));
  CHECK_CUDA_ERROR(cudaFree(d_b));
  CHECK_CUDA_ERROR(cudaFree(d_c));

  elementwise_add_cpu(a, b, cpu_c, len);

  compare(gpu_c, cpu_c, len);

  return 0;
}