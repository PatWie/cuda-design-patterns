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

#if __CUDACC__

#include "include/multiply/multiply.h"

#include "include/cuda_utils.h"

template <typename ValueT>
struct Multiply<ValueT, GpuDevice> {
  static void Apply(const ValueT* A, const ValueT* B, const int H, const int W,
                    ValueT* C) {
    const int num_bytes = H * W * sizeof(ValueT);

    ValueT* d_A;
    ValueT* d_B;
    ValueT* d_C;

    ASSERT_CUDA(cudaMalloc(reinterpret_cast<void**>(&d_A), num_bytes));
    ASSERT_CUDA(cudaMalloc(reinterpret_cast<void**>(&d_B), num_bytes));
    ASSERT_CUDA(cudaMalloc(reinterpret_cast<void**>(&d_C), num_bytes));

    ASSERT_CUDA(cudaMemcpy(d_A, A, num_bytes, cudaMemcpyHostToDevice));
    ASSERT_CUDA(cudaMemcpy(d_B, B, num_bytes, cudaMemcpyHostToDevice));

    multiply_kernels::Multiply<ValueT, 32> kernel;
    kernel.H = H;
    kernel.W = W;
    kernel.A = d_A;
    kernel.B = d_B;
    kernel.C = d_C;
    kernel.Launch();
    ASSERT_CUDA(cudaDeviceSynchronize());

    ASSERT_CUDA(cudaMemcpy(C, d_C, num_bytes, cudaMemcpyDeviceToHost));

    // needed to wait for CUDA kernel output
    // std::cout << cuda::Benchmark(&kernel) << std::endl;
  }
};

template struct Multiply<double, GpuDevice>;
template struct Multiply<float, GpuDevice>;
template struct Multiply<int, GpuDevice>;

#endif  // __CUDACC__
