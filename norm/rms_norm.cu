#include "error.cuh"
#include "macro.h"

#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <iostream>
#include <ratio>
#include <type_traits>

#define float float
#define WARMUP_ITER 10
#define TEST_ITER 100

using namespace std;

template <typename T> __inline__ __device__ T warp_allreduce_sum(T val) {
#pragma unroll
  for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
    val += __shfl_xor_sync(FULL_MASK, val, offset);
  }

  return val;
}

template <typename T> __inline__ __device__ T block_reduce_sum(T val) {
  __shared__ T s_tmp[32];
  auto lane = threadIdx.x % warpSize;
  auto wid = threadIdx.x / warpSize;

  val = warp_allreduce_sum(val);

  if (lane == 0) {
    s_tmp[wid] = val;
  }

  __syncthreads();

  val = threadIdx.x < blockDim.x / warpSize ? s_tmp[lane] : 0;

  if (wid == 0) {
    val = warp_allreduce_sum(val);
  }

  return val;
}

// N = batch_size * seq_length, K = hidden_size
template <typename T>
__global__ void
rms_norm_kernel_gpu(T *__restrict__ out, const T *__restrict__ input,
                    const T *__restrict__ weight, const float epsilon,
                    const int num_tokens, const int hidden_size) {
  __shared__ T s_variance;
  T variance = 0.0;
  for (auto idx = threadIdx.x; idx < hidden_size; idx += blockDim.x) {
    float x = input[blockIdx.x * hidden_size + idx];
    variance += x * x;
  }

  variance = block_reduce_sum<T>(variance);

  if (threadIdx.x == 0) {
    s_variance = rsqrtf(variance / hidden_size + epsilon);
  }

  __syncthreads();

  for (auto idx = threadIdx.x; idx < hidden_size; idx += blockDim.x) {
    float x = input[blockIdx.x * hidden_size + idx];
    out[blockIdx.x * hidden_size + idx] = s_variance * x * weight[idx];
  }
}

// N = batch_size * seq_length, K = hidden_size
// Due to floating-point numbers not satisfying associative laws,
// its reduce sum will cause precision error !
template <typename T>
__global__ void rms_norm_kernel_gpu_vec4(T *out, const T *input,
                                         const T *weight, const float epsilon,
                                         const int num_tokens,
                                         const int hidden_size) {
  __shared__ T s_variance;
  T variance = 0.0;
  for (auto idx = threadIdx.x; idx < hidden_size / 4; idx += blockDim.x) {
    float4 x = *reinterpret_cast<const float4 *>(
        input + blockIdx.x * hidden_size + idx * 4);
    variance += x.x * x.x;
    variance += x.y * x.y;
    variance += x.w * x.w;
    variance += x.z * x.z;
  }

  variance = block_reduce_sum(variance);

  if (threadIdx.x == 0) {
    s_variance = rsqrtf(variance / hidden_size + epsilon);
  }

  __syncthreads();

  for (auto idx = threadIdx.x; idx < hidden_size / 4; idx += blockDim.x) {
    float4 x = *reinterpret_cast<const float4 *>(
        input + blockIdx.x * hidden_size + idx * 4);
    float4 weight = *reinterpret_cast<const float4 *>(input + idx * 4);
    float4 tmp;
    tmp.x = s_variance * x.x * weight.x;
    tmp.w = s_variance * x.w * weight.w;
    tmp.z = s_variance * x.z * weight.z;
    tmp.y = s_variance * x.y * weight.y;
    *reinterpret_cast<float4 *>(out + blockIdx.x * hidden_size + idx * 4) = tmp;
  }
}

template <typename T>
void rms_norm_kernel_cpu(T *__restrict__ out, const T *__restrict__ input,
                         const T *__restrict__ weight, const float epsilon,
                         const int num_tokens, const int hidden_size) {
  for (int i = 0; i < num_tokens; i++) {
    float variance = 0.0f;
    for (int j = 0; j < hidden_size; j++) {
      variance += input[i * hidden_size + j] * input[i * hidden_size + j];
    }
    variance = rsqrtf(variance / static_cast<float>(hidden_size) + epsilon);

    for (int j = 0; j < hidden_size; j++) {
      out[i * hidden_size + j] =
          input[i * hidden_size + j] * weight[j] * variance;
    }
  }
}

void init_random(float *data, size_t len) {
  for (size_t i = 0; i < len; ++i) {
    // data[i] = static_cast<float>(std::rand()) / static_cast<float>(RAND_MAX);
    data[i] = 0.0001f * data[i];
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

void rms_norm_gpu_1(float *out, const float *input, const float *weight,
                    const float epsilon, const int num_tokens,
                    const int hidden_size) {
  rms_norm_kernel_gpu<<<num_tokens, std::min(hidden_size, 1024)>>>(
      out, input, weight, epsilon, num_tokens, hidden_size);
  CHECK_CUDA_ERROR(cudaGetLastError());
}

void rms_norm_gpu_2(float *out, const float *input, const float *weight,
                    const float epsilon, const int num_tokens,
                    const int hidden_size) {
  assert(hidden_size % 4 == 0);

  rms_norm_kernel_gpu_vec4<<<num_tokens, std::min(hidden_size, 1024)>>>(
      out, input, weight, epsilon, num_tokens, hidden_size);
  CHECK_CUDA_ERROR(cudaGetLastError());
}

int main(int argc, char *argv[]) {
  if (argc < 3) {
    std::cout << "[Useage] " << argv[0] << " "
              << "<num tokens> <hidden size>" << std::endl;
    std::exit(1);
  }

  std::srand(std::time(nullptr));

  size_t num_tokens = atol(argv[1]);
  size_t hidden_size = atol(argv[2]);
  const float epsilon = 1e-5;

  float *input = (float *)malloc(num_tokens * hidden_size * sizeof(float));
  float *weight = (float *)malloc(hidden_size * sizeof(float));
  float *cpu_out = (float *)malloc(num_tokens * hidden_size * sizeof(float));
  float *gpu_out = (float *)malloc(num_tokens * hidden_size * sizeof(float));

  init_random(input, num_tokens * hidden_size);
  init_random(weight, hidden_size);

  float *d_input, *d_weight, *d_out;
  CHECK_CUDA_ERROR(
      cudaMalloc(&d_input, num_tokens * hidden_size * sizeof(float)));
  CHECK_CUDA_ERROR(cudaMalloc(&d_weight, hidden_size * sizeof(float)));
  CHECK_CUDA_ERROR(
      cudaMalloc(&d_out, num_tokens * hidden_size * sizeof(float)));

  CHECK_CUDA_ERROR(cudaMemcpy(d_input, input,
                              num_tokens * hidden_size * sizeof(float),
                              cudaMemcpyHostToDevice));
  CHECK_CUDA_ERROR(cudaMemcpy(d_weight, weight, hidden_size * sizeof(float),
                              cudaMemcpyHostToDevice));

  for (int i = 0; i < WARMUP_ITER; ++i) {
    rms_norm_kernel_gpu<<<num_tokens, std::min(hidden_size, 1024UL)>>>(
        d_out, d_input, d_weight, epsilon, num_tokens, hidden_size);
  }

  BENCHMARK(rms_norm_gpu_1, TEST_ITER, d_out, d_input, d_weight, epsilon,
            num_tokens, hidden_size);

  CHECK_CUDA_ERROR(cudaMemcpy(gpu_out, d_out,
                              num_tokens * hidden_size * sizeof(float),
                              cudaMemcpyDeviceToHost));
  rms_norm_kernel_cpu<float>(cpu_out, input, weight, epsilon, num_tokens,
                             hidden_size);
  compare(gpu_out, cpu_out, num_tokens * hidden_size);

  BENCHMARK(rms_norm_gpu_2, TEST_ITER, d_out, d_input, d_weight, epsilon,
            num_tokens, hidden_size);

  CHECK_CUDA_ERROR(cudaMemcpy(gpu_out, d_out,
                              num_tokens * hidden_size * sizeof(float),
                              cudaMemcpyDeviceToHost));
  rms_norm_kernel_cpu<float>(cpu_out, input, weight, epsilon, num_tokens,
                             hidden_size);
  compare(gpu_out, cpu_out, num_tokens * hidden_size);

  return 0;
}