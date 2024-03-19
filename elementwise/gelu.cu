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

void gelu_kernel_cpu(const float *x, float *y, size_t len) {
  for (int i = 0; i < len; ++i) {
    y[i] =
        0.5f * x[i] *
        (1 + tanhf(sqrtf(2 / M_PI) * (x[i] + 0.044715f * x[i] * x[i] * x[i])));
  }
}

__global__ void gelu_kernel_gpu_naive(const float *x, float *y, size_t len) {
  CUDA_KERNEL_LOOP(i, len) {
    y[i] =
        0.5f * x[i] *
        (1 + tanhf(sqrtf(2 / M_PI) * (x[i] + 0.044715f * x[i] * x[i] * x[i])));
  }
}

__global__ void gelu_kernel_gpu_opt1(const float *__restrict__ x,
                                     float *__restrict__ y, size_t len) {
  auto idx = blockDim.x * blockIdx.x + threadIdx.x;
#pragma unroll
  for (auto i = idx; i < len / 4; i += gridDim.x * blockDim.x) {
    float4 val = CONST_FLOAT4(x)[i];
    val.x = 0.5f * val.x *
            (1 + tanhf(sqrtf(2 / M_PI) *
                       (val.x + 0.044715f * val.x * val.x * val.x)));
    ;
    val.y = 0.5f * val.y *
            (1 + tanhf(sqrtf(2 / M_PI) *
                       (val.y + 0.044715f * val.y * val.y * val.y)));
    ;
    val.z = 0.5f * val.z *
            (1 + tanhf(sqrtf(2 / M_PI) *
                       (val.z + 0.044715f * val.z * val.z * val.z)));
    ;
    val.w = 0.5f * val.w *
            (1 + tanhf(sqrtf(2 / M_PI) *
                       (val.w + 0.044715f * val.w * val.w * val.w)));
    ;
    FLOAT4(y)[i] = val;
  }
  auto remainder = len % 4;
  if (idx / 4 == 0 && remainder != 0) {
    while (remainder) {
      auto idx = len - remainder--;
      y[idx] = 0.5f * x[idx] *
               (1 + tanhf(sqrtf(2 / M_PI) *
                          (x[idx] + 0.044715f * x[idx] * x[idx] * x[idx])));
    }
  }
}

__global__ void gelu_kernel_gpu_opt2(const float *__restrict__ x,
                                     float *__restrict__ y, size_t len) {
  auto idx = blockDim.x * blockIdx.x + threadIdx.x;
  constexpr float R_M_PIx2 = 2 / M_PI;
#pragma unroll
  for (auto i = idx; i < len / 4; i += gridDim.x * blockDim.x) {
    float4 val = CONST_FLOAT4(x)[i];
    float tx = val.x + (0.044715f * val.x) * (val.x * val.x);
    float ty = val.y + (0.044715f * val.y) * (val.y * val.y);
    val.x = (0.5f * val.x) * (1 + tanhf(__fsqrt_rn(R_M_PIx2) * tx));
    val.y = (0.5f * val.y) * (1 + tanhf(__fsqrt_rn(R_M_PIx2) * ty));
    float tz = val.z + (0.044715f * val.z) * (val.z * val.z);
    float tw = val.w + (0.044715f * val.w) * (val.w * val.w);
    val.z = (0.5f * val.z) * (1 + tanhf(__fsqrt_rn(R_M_PIx2) * tz));
    val.w = (0.5f * val.w) * (1 + tanhf(__fsqrt_rn(R_M_PIx2) * tw));
    FLOAT4(y)[i] = val;
  }
  auto remainder = len % 4;
  if (idx / 4 == 0 && remainder != 0) {
    while (remainder) {
      auto idx = len - remainder--;
      float t = (x[idx] + 0.044715f * x[idx] * x[idx] * x[idx]);
      y[idx] = 0.5f * x[idx] * (1 + tanhf(__fsqrt_rn(R_M_PIx2) * t));
    }
  }
}

void init_random(float *data, size_t len) {
  for (size_t i = 0; i < len; ++i) {
    data[i] = static_cast<float>(std::rand()) / static_cast<float>(RAND_MAX);
  }
}

void gelu_gpu_naive(const float *x, float *y, size_t len) {
  gelu_kernel_gpu_naive<<<BLOCKS_PER_GRID(len), THREADS_PER_BLOCK>>>(x, y, len);
  cudaStreamSynchronize(0);
  CHECK_CUDA_ERROR(cudaGetLastError());
}

void gelu_gpu_1(const float *x, float *y, size_t len) {
  gelu_kernel_gpu_opt1<<<BLOCKS_PER_GRID(len) / 4, THREADS_PER_BLOCK>>>(x, y,
                                                                        len);
  cudaStreamSynchronize(0);
  CHECK_CUDA_ERROR(cudaGetLastError());
}

void gelu_gpu_2(const float *x, float *y, size_t len) {
  gelu_kernel_gpu_opt2<<<BLOCKS_PER_GRID(len) / 4, THREADS_PER_BLOCK>>>(x, y,
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
    gelu_kernel_gpu_naive<<<BLOCKS_PER_GRID(len), THREADS_PER_BLOCK>>>(d_a, d_b,
                                                                       len);
  }

  gelu_kernel_cpu(a, cpu_b, len);

  gelu_gpu_naive(d_a, d_b, len);
  CHECK_CUDA_ERROR(cudaMemcpy(gpu_b, d_b, size, cudaMemcpyDeviceToHost));
  compare(gpu_b, cpu_b, len);

  gelu_gpu_1(d_a, d_b, len);
  CHECK_CUDA_ERROR(cudaMemcpy(gpu_b, d_b, size, cudaMemcpyDeviceToHost));
  compare(gpu_b, cpu_b, len);

  gelu_gpu_2(d_a, d_b, len);
  CHECK_CUDA_ERROR(cudaMemcpy(gpu_b, d_b, size, cudaMemcpyDeviceToHost));
  compare(gpu_b, cpu_b, len);

  CHECK_CUDA_ERROR(cudaFree(d_a));
  CHECK_CUDA_ERROR(cudaFree(d_b));

  return 0;
}