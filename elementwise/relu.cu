#include "error.cuh"
#include "macro.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <iostream>
#include <math.h>
#include <type_traits>

#define DATATYPE float
#define WARMUP_ITER 10
#define TEST_ITER 100

using namespace std;

void relu_cpu(const DATATYPE *x, DATATYPE *y, size_t len) {
  for (int i = 0; i < len; ++i) {
    y[i] = std::fmax(x[i], 0);
  }
}

__global__ void relu_gpu_naive(const DATATYPE *x, DATATYPE *y, size_t len) {
  CUDA_KERNEL_LOOP(i, len) { y[i] = x[i] > 0 ? x[i] : 0; }
}

__global__ void relu_gpu_opt1(const DATATYPE *x, DATATYPE *y, size_t len) {
  CUDA_KERNEL_LOOP(i, len) { y[i] = fmaxf(x[i], 0.0f); }
}

__global__ void relu_gpu_opt2(const DATATYPE *x, DATATYPE *y, size_t len) {
  auto idx = blockDim.x * blockIdx.x + threadIdx.x;
#pragma unroll
  for (int i = idx; i < len / 4; i += gridDim.x * blockDim.x) {
    float4 val = CONST_FLOAT4(x)[i];
    val.x = fmaxf(val.x, 0);
    val.y = fmaxf(val.y, 0);
    val.z = fmaxf(val.z, 0);
    val.w = fmaxf(val.w, 0);
    FLOAT4(y)[i] = val;
  }
  int remainder = len % 4;
  if (idx / 4 == 0 && remainder != 0) {
    while (remainder) {
      int idx = len - remainder--;
      y[idx] = fmaxf(x[idx], 0);
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
  DATATYPE *cpu_b = (DATATYPE *)malloc(size);
  memset(cpu_b, 0, sizeof(DATATYPE) * len);
  DATATYPE *gpu_b = (DATATYPE *)malloc(size);
  memset(gpu_b, 0, sizeof(DATATYPE) * len);

  init_random(a, len);
  init_random(b, len);

  DATATYPE *d_a, *d_b;
  CHECK_CUDA_ERROR(cudaMalloc(&d_a, size));
  CHECK_CUDA_ERROR(cudaMalloc(&d_b, size));

  CHECK_CUDA_ERROR(cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice));

  for (int i = 0; i < WARMUP_ITER; ++i) {
    relu_gpu_naive<<<BLOCKS_PER_GRID(len), THREADS_PER_BLOCK>>>(d_a, d_b, len);
  }

  for (int i = 0; i < TEST_ITER; ++i) {
    relu_gpu_naive<<<BLOCKS_PER_GRID(len), THREADS_PER_BLOCK>>>(d_a, d_b, len);
  }

  for (int i = 0; i < TEST_ITER; ++i) {
    relu_gpu_opt1<<<BLOCKS_PER_GRID(len), THREADS_PER_BLOCK>>>(d_a, d_b, len);
  }

  for (int i = 0; i < TEST_ITER; ++i) {
    relu_gpu_opt2<<<BLOCKS_PER_GRID(len) / 4, THREADS_PER_BLOCK>>>(d_a, d_b,
                                                                   len);
  }

  CHECK_CUDA_ERROR(cudaGetLastError());

  CHECK_CUDA_ERROR(cudaMemcpy(gpu_b, d_b, size, cudaMemcpyDeviceToHost));

  CHECK_CUDA_ERROR(cudaFree(d_a));
  CHECK_CUDA_ERROR(cudaFree(d_b));

  relu_cpu(a, cpu_b, len);

  compare(gpu_b, cpu_b, len);

  return 0;
}