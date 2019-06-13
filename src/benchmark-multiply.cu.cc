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

#include "include/cuda_benchmark.h"
#include "include/cuda_utils.h"
#include "include/multiply/multiply.h"

struct Initializer {
  explicit Initializer(float *d_A, float *d_B, float *d_C, int height,
                       int width)
      : d_A(d_A), d_B(d_B), d_C(d_C), height(height), width(width) {}

  template <typename TKernel>
  void operator()(TKernel *kernel) {
    kernel->H = height;
    kernel->W = width;
    kernel->A = d_A;
    kernel->B = d_B;
    kernel->C = d_C;
  }

  float *d_A;
  float *d_B;
  float *d_C;
  int height;
  int width;
};

// Simple way to benchmark the template parameters.
void run_for(int height, int width) {
  std::cout << std::endl
            << "Benchmark for " << height << " " << width
            << " -------------------------------- " << std::endl;

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

  Initializer init(d_A, d_B, d_C, height, width);

  // Test different options.
  cuda::KernelBenchmarker<int> bench;
  bench.Case<multiply_kernels::Multiply<float, 2> >(init);
  bench.Case<multiply_kernels::Multiply<float, 3> >(init);
  bench.Case<multiply_kernels::Multiply<float, 4> >(init);
  bench.Case<multiply_kernels::Multiply<float, 6> >(init);
  bench.Case<multiply_kernels::Multiply<float, 8> >(init);
  bench.Case<multiply_kernels::Multiply<float, 10> >(init);
  bench.Case<multiply_kernels::Multiply<float, 16> >(init);
  bench.Case<multiply_kernels::Multiply<float, 20> >(init);
  bench.Case<multiply_kernels::Multiply<float, 32> >(init);
  bench.Run();

  delete[] A;
  delete[] B;
  delete[] C;
  ASSERT_CUDA(cudaFree(d_A));
  ASSERT_CUDA(cudaFree(d_B));
  ASSERT_CUDA(cudaFree(d_C));
}

int main() {
  run_for(256, 256);
  run_for(512, 512);
  run_for(1024, 1024);
  return 0;
}

/*
Output:
Benchmark for 256 256 --------------------------------
multiply_kernels::Multiply<float, 2>  took 2.91322 ms
multiply_kernels::Multiply<float, 3>  took 0.956768 ms
multiply_kernels::Multiply<float, 4>  took 0.397472 ms
multiply_kernels::Multiply<float, 6>  took 0.193184 ms
multiply_kernels::Multiply<float, 8>  took 0.096512 ms
multiply_kernels::Multiply<float, 10>  took 0.118912 ms
multiply_kernels::Multiply<float, 16>  took 0.084544 ms
multiply_kernels::Multiply<float, 20>  took 0.098304 ms
multiply_kernels::Multiply<float, 32>  took 0.0816 ms

Benchmark for 512 512 --------------------------------
multiply_kernels::Multiply<float, 2>  took 25.7449 ms
multiply_kernels::Multiply<float, 3>  took 8.61693 ms
multiply_kernels::Multiply<float, 4>  took 3.18605 ms
multiply_kernels::Multiply<float, 6>  took 1.42717 ms
multiply_kernels::Multiply<float, 8>  took 0.674464 ms
multiply_kernels::Multiply<float, 10>  took 0.848448 ms
multiply_kernels::Multiply<float, 16>  took 0.590048 ms
multiply_kernels::Multiply<float, 20>  took 0.6624 ms
multiply_kernels::Multiply<float, 32>  took 0.603904 ms

Benchmark for 1024 1024 --------------------------------
multiply_kernels::Multiply<float, 2>  took 245.845 ms
multiply_kernels::Multiply<float, 3>  took 77.8342 ms
multiply_kernels::Multiply<float, 4>  took 33.6046 ms
multiply_kernels::Multiply<float, 6>  took 12.244 ms
multiply_kernels::Multiply<float, 8>  took 5.71571 ms
multiply_kernels::Multiply<float, 10>  took 6.38643 ms
multiply_kernels::Multiply<float, 16>  took 4.60816 ms
multiply_kernels::Multiply<float, 20>  took 5.86672 ms
multiply_kernels::Multiply<float, 32>  took 4.24483 ms

*/
