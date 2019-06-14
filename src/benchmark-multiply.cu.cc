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
  cuda::KernelBenchmark<int> bench;
  bench.Case<multiply_kernels::Multiply<float, 2>>(init);
  bench.Case<multiply_kernels::Multiply<float, 3>>(init);
  bench.Case<multiply_kernels::Multiply<float, 4>>(init);
  bench.Case<multiply_kernels::Multiply<float, 6>>(init);
  bench.Case<multiply_kernels::Multiply<float, 8>>(init);
  bench.Case<multiply_kernels::Multiply<float, 10>>(init);
  bench.Case<multiply_kernels::Multiply<float, 16>>(init);
  bench.Case<multiply_kernels::Multiply<float, 20>>(init);
  bench.Case<multiply_kernels::Multiply<float, 32>>(init);
  bench.Start();

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

// clang-format off
/*
Benchmark for 256 256 --------------------------------
Using Device Number: 0
  Device name: GeForce GTX 970
  Memory Clock Rate (KHz): 3505000
  Memory Bus Width (bits): 256
  Peak Memory Bandwidth (GB/s): 224.320000

time 500.000000 - 1000.000000, iters: 5 - 100
 - multiply_kernels::Multiply<float, 2>    took     3.047784 ms stats(iters: 100, var:     0.082136, stddev:     0.286594)
 - multiply_kernels::Multiply<float, 3>    took     0.847197 ms stats(iters: 100, var:     0.002289, stddev:     0.047846)
 - multiply_kernels::Multiply<float, 4>    took     0.337858 ms stats(iters: 100, var:     0.000039, stddev:     0.006252)
 - multiply_kernels::Multiply<float, 6>    took     0.162206 ms stats(iters: 100, var:     0.000004, stddev:     0.001925)
 - multiply_kernels::Multiply<float, 8>    took     0.081275 ms stats(iters: 100, var:     0.000000, stddev:     0.000677)
 - multiply_kernels::Multiply<float, 10>   took     0.100844 ms stats(iters: 100, var:     0.000000, stddev:     0.000486)
 - multiply_kernels::Multiply<float, 16>   took     0.072184 ms stats(iters: 100, var:     0.000001, stddev:     0.000723)
 - multiply_kernels::Multiply<float, 20>   took     0.082570 ms stats(iters: 100, var:     0.000004, stddev:     0.001894)
 - multiply_kernels::Multiply<float, 32>   took     0.070467 ms stats(iters: 100, var:     0.000008, stddev:     0.002803)

Benchmark for 512 512 --------------------------------
Using Device Number: 0
  Device name: GeForce GTX 970
  Memory Clock Rate (KHz): 3505000
  Memory Bus Width (bits): 256
  Peak Memory Bandwidth (GB/s): 224.320000

time 500.000000 - 1000.000000, iters: 5 - 100
 - multiply_kernels::Multiply<float, 2>    took    20.967186 ms stats(iters:  48, var:     1.166002, stddev:     1.079816)
 - multiply_kernels::Multiply<float, 3>    took     6.682436 ms stats(iters: 100, var:     0.122818, stddev:     0.350454)
 - multiply_kernels::Multiply<float, 4>    took     2.826743 ms stats(iters: 100, var:     0.067757, stddev:     0.260302)
 - multiply_kernels::Multiply<float, 6>    took     1.245100 ms stats(iters: 100, var:     0.019352, stddev:     0.139112)
 - multiply_kernels::Multiply<float, 8>    took     0.574468 ms stats(iters: 100, var:     0.000003, stddev:     0.001616)
 - multiply_kernels::Multiply<float, 10>   took     0.713191 ms stats(iters: 100, var:     0.000003, stddev:     0.001810)
 - multiply_kernels::Multiply<float, 16>   took     0.502195 ms stats(iters: 100, var:     0.000002, stddev:     0.001380)
 - multiply_kernels::Multiply<float, 20>   took     0.560309 ms stats(iters: 100, var:     0.000006, stddev:     0.002414)
 - multiply_kernels::Multiply<float, 32>   took     0.510635 ms stats(iters: 100, var:     0.000001, stddev:     0.001121)

Benchmark for 1024 1024 --------------------------------
Using Device Number: 0
  Device name: GeForce GTX 970
  Memory Clock Rate (KHz): 3505000
  Memory Bus Width (bits): 256
  Peak Memory Bandwidth (GB/s): 224.320000

time 500.000000 - 1000.000000, iters: 5 - 100
 - multiply_kernels::Multiply<float, 2>    took   287.646912 ms stats(iters:   4, var:   126.933113, stddev:    11.266459)
 - multiply_kernels::Multiply<float, 3>    took    78.918053 ms stats(iters:  13, var:     1.950417, stddev:     1.396573)
 - multiply_kernels::Multiply<float, 4>    took    33.681572 ms stats(iters:  15, var:     0.029435, stddev:     0.171566)
 - multiply_kernels::Multiply<float, 6>    took    12.483257 ms stats(iters:  41, var:     0.002221, stddev:     0.047123)
 - multiply_kernels::Multiply<float, 8>    took     5.562872 ms stats(iters: 100, var:     0.034724, stddev:     0.186343)
 - multiply_kernels::Multiply<float, 10>   took     6.286970 ms stats(iters: 100, var:     0.010179, stddev:     0.100893)
 - multiply_kernels::Multiply<float, 16>   took     4.158412 ms stats(iters: 100, var:     0.043726, stddev:     0.209108)
 - multiply_kernels::Multiply<float, 20>   took     4.711136 ms stats(iters: 100, var:     0.064436, stddev:     0.253843)
 - multiply_kernels::Multiply<float, 32>   took     4.059203 ms stats(iters: 100, var:     0.044180, stddev:     0.210191)

*/
// clang-format on
