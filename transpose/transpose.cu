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

#define TILE_DIM 32

void transpose_kernel_cpu(const float *a, float *b, const int m, const int n) {
  for (int i = 0; i < m; i++) {
    for (int j = 0; j < n; j++) {
      b[i * n + j] = a[j * n + i];
    }
  }
}

__global__ void transpose_kernel_gpu_1(const float *__restrict__ a,
                                       float *__restrict__ b, const int m,
                                       const int n) {
  const auto x = TILE_DIM * blockIdx.x + threadIdx.x;
  const auto y = TILE_DIM * blockIdx.y + threadIdx.y;

  if (y < m && x < n) {
    // read coalesing
    b[x * m + y] = a[y * n + x];
  }
}

__global__ void transpose_kernel_gpu_2(const float *__restrict__ a,
                                       float *__restrict__ b, const int m,
                                       const int n) {
  const auto x = TILE_DIM * blockIdx.x + threadIdx.x;
  const auto y = TILE_DIM * blockIdx.y + threadIdx.y;

  if (y < m && x < n) {
    // write coalesing
    b[y * n + x] = __ldg(&a[x * m + y]);
  }
}

__global__ void transpose_kernel_gpu_3(const float *__restrict__ a,
                                       float *__restrict__ b, const int m,
                                       const int n) {
  const auto bx = blockIdx.x * TILE_DIM;
  const auto by = blockIdx.y * TILE_DIM;

  __shared__ float s_tile[TILE_DIM][TILE_DIM];

  const auto nx1 = bx + threadIdx.x;
  const auto ny1 = by + threadIdx.y;
  if (ny1 < m && nx1 < n) {
    s_tile[threadIdx.y][threadIdx.x] = a[ny1 * n + nx1];
  }

  __syncthreads();

  // trans block, keep thread continous
  const auto nx2 = by + threadIdx.x;
  const auto ny2 = bx + threadIdx.y;
  if (ny2 < m && nx2 < n) {
    // write coalesing
    // it will cause bank conflict, eg. thread0 will read 0, 31, 63... there are
    // in same bank.
    b[ny2 * n + nx2] = s_tile[threadIdx.x][threadIdx.y];
  }
}

__global__ void transpose_kernel_gpu_4(const float *__restrict__ a,
                                       float *__restrict__ b, const int m,
                                       const int n) {
  const auto bx = blockIdx.x * TILE_DIM;
  const auto by = blockIdx.y * TILE_DIM;

  // use zero padding to avoid bank conflict
  __shared__ float s_tile[TILE_DIM][TILE_DIM + 1];

  const auto nx1 = bx + threadIdx.x;
  const auto ny1 = by + threadIdx.y;
  if (ny1 < m && nx1 < n) {
    s_tile[threadIdx.y][threadIdx.x] = a[ny1 * n + nx1];
  }

  __syncthreads();

  // trans block, keep thread continous
  const auto nx2 = by + threadIdx.x;
  const auto ny2 = bx + threadIdx.y;
  if (ny2 < m && nx2 < n) {
    // write coalesing
    b[ny2 * n + nx2] = s_tile[threadIdx.x][threadIdx.y];
  }
}

void transpose_gpu_1(const float *a, float *b, const int m, const int n) {
  dim3 grid_dim((n + TILE_DIM - 1) / TILE_DIM, (m + TILE_DIM - 1) / TILE_DIM);
  dim3 block_dim(TILE_DIM, TILE_DIM);

  transpose_kernel_gpu_1<<<grid_dim, block_dim>>>(a, b, m, n);
  cudaStreamSynchronize(0);
  CHECK_CUDA_ERROR(cudaGetLastError());
}

void transpose_gpu_2(const float *a, float *b, const int m, const int n) {
  dim3 grid_dim((n + TILE_DIM - 1) / TILE_DIM, (m + TILE_DIM - 1) / TILE_DIM);
  dim3 block_dim(TILE_DIM, TILE_DIM);

  transpose_kernel_gpu_2<<<grid_dim, block_dim>>>(a, b, m, n);
  cudaStreamSynchronize(0);
  CHECK_CUDA_ERROR(cudaGetLastError());
}

void transpose_gpu_3(const float *a, float *b, const int m, const int n) {
  dim3 grid_dim((n + TILE_DIM - 1) / TILE_DIM, (m + TILE_DIM - 1) / TILE_DIM);
  dim3 block_dim(TILE_DIM, TILE_DIM);

  transpose_kernel_gpu_3<<<grid_dim, block_dim>>>(a, b, m, n);
  cudaStreamSynchronize(0);
  CHECK_CUDA_ERROR(cudaGetLastError());
}

void transpose_gpu_4(const float *a, float *b, const int m, const int n) {
  dim3 grid_dim((n + TILE_DIM - 1) / TILE_DIM, (m + TILE_DIM - 1) / TILE_DIM);
  dim3 block_dim(TILE_DIM, TILE_DIM);

  transpose_kernel_gpu_4<<<grid_dim, block_dim>>>(a, b, m, n);
  cudaStreamSynchronize(0);
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
  if (argc < 3) {
    std::cout << "[Useage] " << argv[0] << " "
              << "<m> <n>" << std::endl;
    std::exit(1);
  }
  int num_sm;
  cudaDeviceGetAttribute(&num_sm, cudaDevAttrMultiProcessorCount, 0);

  std::srand(std::time(nullptr));

  size_t m = atol(argv[1]);
  size_t n = atol(argv[2]);
  size_t size = m * n * sizeof(float);

  float *a = (float *)malloc(size);
  float *gpu_b = (float *)malloc(size);
  float *cpu_b = (float *)malloc(size);

  init_random(a, m * n);

  float *d_a, *d_b;
  CHECK_CUDA_ERROR(cudaMalloc(&d_a, size));
  CHECK_CUDA_ERROR(cudaMalloc(&d_b, size));

  CHECK_CUDA_ERROR(cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice));

  transpose_kernel_cpu(a, cpu_b, m, n);

  for (int i = 0; i < WARMUP_ITER; ++i) {
    dim3 grid_dim((n + TILE_DIM - 1) / TILE_DIM, (m + TILE_DIM - 1) / TILE_DIM);
    dim3 block_dim(TILE_DIM, TILE_DIM);

    transpose_kernel_gpu_1<<<grid_dim, block_dim>>>(d_a, d_b, m, n);
  }

  transpose_gpu_1(d_a, d_b, m, n);
  CHECK_CUDA_ERROR(cudaMemcpy(gpu_b, d_b, size, cudaMemcpyDeviceToHost));
  compare(gpu_b, cpu_b, m * n);

  transpose_gpu_2(d_a, d_b, m, n);
  CHECK_CUDA_ERROR(cudaMemcpy(gpu_b, d_b, size, cudaMemcpyDeviceToHost));
  compare(gpu_b, cpu_b, m * n);

  transpose_gpu_3(d_a, d_b, m, n);
  CHECK_CUDA_ERROR(cudaMemcpy(gpu_b, d_b, size, cudaMemcpyDeviceToHost));
  compare(gpu_b, cpu_b, m * n);

  transpose_gpu_4(d_a, d_b, m, n);
  CHECK_CUDA_ERROR(cudaMemcpy(gpu_b, d_b, size, cudaMemcpyDeviceToHost));
  compare(gpu_b, cpu_b, m * n);

  CHECK_CUDA_ERROR(cudaFree(d_a));
  CHECK_CUDA_ERROR(cudaFree(d_b));

  return 0;
}