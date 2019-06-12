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

#ifndef INCLUDE_MULTIPLY_MULTIPLY_H_
#define INCLUDE_MULTIPLY_MULTIPLY_H_

#include "include/cuda_index.h"
#include "include/cuda_utils.h"

#if __CUDACC__
namespace multiply_kernels {

using cuda::make_ndarray;

// We follow the NVIDIA cub library style for template parameters.
// BLOCK_DIM_X is the number of threads in a block along dimension x.
template <typename ValueT, int BLOCK_DIM_X>
struct Multiply : public cuda::Kernel {
  // use enum for compiletime config
  // enum { PER_THREAD = 1 };

  void Launch(cudaStream_t stream = 0) override {
    dim3 block(BLOCK_DIM_X, BLOCK_DIM_X);
    dim3 grid(cuda::divUp(W, BLOCK_DIM_X), cuda::divUp(H, BLOCK_DIM_X));

    cuda::Run<<<grid, block, 0, stream>>>(*this);
  }

  __device__ __forceinline__ void operator()() const override {
    __shared__ ValueT ds_M[BLOCK_DIM_X][BLOCK_DIM_X];
    __shared__ ValueT ds_N[BLOCK_DIM_X][BLOCK_DIM_X];

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int Ch = blockIdx.y * BLOCK_DIM_X + ty;
    const int Cw = blockIdx.x * BLOCK_DIM_X + tx;

    ValueT Cval = 0;

    const auto At = make_ndarray<const ValueT, 2>(A, H, W);
    const auto Bt = make_ndarray<const ValueT, 2>(B, H, W);
    auto Ct = make_ndarray<ValueT, 2>(C, H, W);

    for (int m = 0; m < (W - 1) / BLOCK_DIM_X + 1; ++m) {
      if (At.valid(Ch, m * BLOCK_DIM_X + tx)) {
        ds_M[ty][tx] = At(Ch, m * BLOCK_DIM_X + tx);
      } else {
        ds_N[ty][tx] = 0;
      }
      if (Bt.valid(m * BLOCK_DIM_X + ty, Cw)) {
        ds_N[ty][tx] = Bt(m * BLOCK_DIM_X + ty, Cw);
      } else {
        ds_N[ty][tx] = 0;
      }
      __syncthreads();

      for (int k = 0; k < BLOCK_DIM_X; ++k) Cval += ds_M[ty][k] * ds_N[k][tx];
      __syncthreads();
    }
    if (Ct.valid(Ch, Cw)) Ct(Ch, Cw) = Cval;
  }

  int W;
  int H;
  const ValueT* A;
  const ValueT* B;
  ValueT* C;
};

}  // namespace multiply_kernels
#endif  // __CUDACC__

template <typename ValueT, typename Device = AnyDevice>
struct Multiply {
  static void Apply(const ValueT* A, const ValueT* B, const int H, const int W,
                    ValueT* C);
};

#endif  // INCLUDE_MULTIPLY_MULTIPLY_H_
