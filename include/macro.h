#ifndef __MACRO_H__
#define __MACRO_H__

#include <chrono>

#define MAX(a, b) ((a > b) ? a : b)
#define MIN(a, b) ((a < b) ? a : b)

#define CONST_FLOAT4(pointer) (reinterpret_cast<const float4 *>(pointer))
#define FLOAT4(pointer) (reinterpret_cast<float4 *>(pointer))

#define FULL_MASK 0xffffffff

#define MAX_BLOCKS 65536
#define THREADS_PER_BLOCK 128
#define BLOCKS_PER_GRID(n)                                                     \
  (MIN((((n) + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK), MAX_BLOCKS))

// CUDA: grid stride looping
#define CUDA_KERNEL_LOOP(i, n)                                                 \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < (n);                 \
       i += blockDim.x * gridDim.x)

#define BENCHMARK(func, times, args...)                                        \
  {                                                                            \
    auto begin = std::chrono::steady_clock::now();                             \
    for (int i = 0; i < times; ++i) {                                          \
      func(args);                                                              \
    }                                                                          \
    auto end = std::chrono::steady_clock::now();                               \
    std::cout                                                                  \
        << #func " elapsed time: "                                             \
        << std::chrono::duration<double, std::milli>(end - begin).count()      \
        << "ms" << std::endl;                                                  \
  }

#endif