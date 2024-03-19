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

#define WARMUP_ITER 10
#define TEST_ITER 100

using namespace std;

void relu_kernel_cpu(const float *x, float *y, size_t len) {
  for (int i = 0; i < len; ++i) {
    y[i] = std::fmax(x[i], 0);
  }
}

__global__ void relu_kernel_gpu_naive(const float *x, float *y, size_t len) {
  CUDA_KERNEL_LOOP(i, len) { y[i] = x[i] > 0 ? x[i] : 0; }
}

__global__ void relu_kernel_gpu_opt1(const float *x, float *y, size_t len) {
  CUDA_KERNEL_LOOP(i, len) { y[i] = fmaxf(x[i], 0.0f); }
}

__global__ void relu_kernel_gpu_opt2(const float *x, float *y, size_t len) {
  auto idx = blockDim.x * blockIdx.x + threadIdx.x;
#pragma unroll
  for (auto i = idx; i < len / 4; i += gridDim.x * blockDim.x) {
    float4 val = CONST_FLOAT4(x)[i];
    val.x = fmaxf(val.x, 0);
    val.y = fmaxf(val.y, 0);
    val.z = fmaxf(val.z, 0);
    val.w = fmaxf(val.w, 0);
    FLOAT4(y)[i] = val;
  }
  auto remainder = len % 4;
  if (idx / 4 == 0 && remainder != 0) {
    while (remainder) {
      auto idx = len - remainder--;
      y[idx] = fmaxf(x[idx], 0);
    }
  }
}

void init_random(float *data, size_t len) {
  for (size_t i = 0; i < len; ++i) {
    data[i] = static_cast<float>(std::rand()) / static_cast<float>(RAND_MAX);
  }
}

void relu_gpu_naive(const float *x, float *y, size_t len) {
  relu_kernel_gpu_naive<<<BLOCKS_PER_GRID(len), THREADS_PER_BLOCK>>>(x, y, len);
  cudaStreamSynchronize(0);
  CHECK_CUDA_ERROR(cudaGetLastError());
}

void relu_gpu_1(const float *x, float *y, size_t len) {
  relu_kernel_gpu_opt1<<<BLOCKS_PER_GRID(len), THREADS_PER_BLOCK>>>(x, y, len);
  cudaStreamSynchronize(0);
  CHECK_CUDA_ERROR(cudaGetLastError());
}
void relu_gpu_2(const float *x, float *y, size_t len) {
  relu_kernel_gpu_opt2<<<BLOCKS_PER_GRID(len) / 4, THREADS_PER_BLOCK>>>(x, y,
                                                                        len);
  cudaStreamSynchronize(0);
  CHECK_CUDA_ERROR(cudaGetLastError());
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
  float *cpu_b = (float *)malloc(size);
  memset(cpu_b, 0, sizeof(float) * len);
  float *gpu_b = (float *)malloc(size);
  memset(gpu_b, 0, sizeof(float) * len);

  init_random(a, len);
  init_random(b, len);

  float *d_a, *d_b;
  CHECK_CUDA_ERROR(cudaMalloc(&d_a, size));
  CHECK_CUDA_ERROR(cudaMalloc(&d_b, size));

  CHECK_CUDA_ERROR(cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice));

  for (int i = 0; i < WARMUP_ITER; ++i) {
    relu_kernel_gpu_naive<<<BLOCKS_PER_GRID(len), THREADS_PER_BLOCK>>>(d_a, d_b,
                                                                       len);
  }

  relu_kernel_cpu(a, cpu_b, len);

  relu_gpu_naive(d_a, d_b, len);
  CHECK_CUDA_ERROR(cudaMemcpy(gpu_b, d_b, size, cudaMemcpyDeviceToHost));
  compare(gpu_b, cpu_b, len);

  relu_gpu_1(d_a, d_b, len);
  CHECK_CUDA_ERROR(cudaMemcpy(gpu_b, d_b, size, cudaMemcpyDeviceToHost));
  compare(gpu_b, cpu_b, len);

  relu_gpu_2(d_a, d_b, len);
  CHECK_CUDA_ERROR(cudaMemcpy(gpu_b, d_b, size, cudaMemcpyDeviceToHost));
  compare(gpu_b, cpu_b, len);

  CHECK_CUDA_ERROR(cudaFree(d_a));
  CHECK_CUDA_ERROR(cudaFree(d_b));

  return 0;
}