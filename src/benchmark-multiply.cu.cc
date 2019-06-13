/* Copyright 2019 Authors. All Rights Reserved.

 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * Author: Patrick Wieschollek, <mail@patwie.com>, 2019
 *
 */

#include "include/cuda_utils.h"
#include "include/multiply/multiply.h"

struct Initializer {
  explicit Initializer(float *d_A, float *d_B, float *d_C)
      : d_A(d_A), d_B(d_B), d_C(d_C) {}

  template <typename TKernel>
  void operator()(TKernel *kernel) {
    kernel->H = 1024;
    kernel->W = 1024;
    kernel->A = d_A;
    kernel->B = d_B;
    kernel->C = d_C;
  }

  float *d_A;
  float *d_B;
  float *d_C;
};

// Simple way to benchmark the template parameters.
int main() {
  constexpr int height = 1024;
  constexpr int width = 1024;
  float *A = new float[height * width];
  float *B = new float[height * width];
  float *C = new float[height * width];
  for (int i = 0; i < height * width; ++i) {
    A[i] = i;
    B[i] = i * 10;
    C[i] = 0;
  }

  float *d_A;
  float *d_B;
  float *d_C;

  const int num_bytes = height * width * sizeof(float);

  ASSERT_CUDA(cudaMalloc(reinterpret_cast<void **>(&d_A), num_bytes));
  ASSERT_CUDA(cudaMalloc(reinterpret_cast<void **>(&d_B), num_bytes));
  ASSERT_CUDA(cudaMalloc(reinterpret_cast<void **>(&d_C), num_bytes));

  ASSERT_CUDA(cudaMemcpy(d_A, A, num_bytes, cudaMemcpyHostToDevice));
  ASSERT_CUDA(cudaMemcpy(d_B, B, num_bytes, cudaMemcpyHostToDevice));

  Initializer init(d_A, d_B, d_C);

  // test different options
  cuda::KernelDispatcher<int> bench;
  bench.Register<multiply_kernels::Multiply<float, 2> >(2, init);
  bench.Register<multiply_kernels::Multiply<float, 3> >(3, init);
  bench.Register<multiply_kernels::Multiply<float, 4> >(4, init);
  bench.Register<multiply_kernels::Multiply<float, 6> >(6, init);
  bench.Register<multiply_kernels::Multiply<float, 8> >(8, init);
  bench.Register<multiply_kernels::Multiply<float, 10> >(10, init);
  bench.Register<multiply_kernels::Multiply<float, 16> >(16, init);
  bench.Register<multiply_kernels::Multiply<float, 20> >(20, init);
  bench.Register<multiply_kernels::Multiply<float, 32> >(32, init);

  bench.Benchmark();

  return 0;
}

/*
Output:
key 2 [multiply_kernels::Multiply<float, 2>] ... took 250.052 ms
key 3 [multiply_kernels::Multiply<float, 3>] ... took 77.2038 ms
key 4 [multiply_kernels::Multiply<float, 4>] ... took 35.8959 ms
key 6 [multiply_kernels::Multiply<float, 6>] ... took 12.6762 ms
key 8 [multiply_kernels::Multiply<float, 8>] ... took 5.67414 ms
key 10 [multiply_kernels::Multiply<float, 10>] ... took 6.18115 ms
key 16 [multiply_kernels::Multiply<float, 16>] ... took 4.0704 ms
key 20 [multiply_kernels::Multiply<float, 20>] ... took 6.8576 ms
key 32 [multiply_kernels::Multiply<float, 32>] ... took 4.73555 ms

*/
