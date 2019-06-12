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

namespace {

// Example for 1D generic
template <typename ValueT, int BLOCK_DIM_X>
struct ExpertKernel1D : public cuda::Kernel {
  void Launch(cudaStream_t stream = 0) override {
    dim3 block(BLOCK_DIM_X);
    dim3 grid(1);
    cuda::Run<<<grid, block, 0, stream>>>(*this);
  }

  __device__ __forceinline__ void operator()() const override {
    printf("thread %d here from the expert-kernel %d [val=%f]\n", threadIdx.x,
           BLOCK_DIM_X, val);
  }

  ValueT val = 0;
};

// Example for 1D special
template <typename ValueT>
struct ExpertKernel1D<ValueT, 4> : public cuda::Kernel {
  void Launch(cudaStream_t stream = 0) override {
    dim3 block(4);
    dim3 grid(1);
    cuda::Run<<<grid, block, 0, stream>>>(*this);
  }

  __device__ __forceinline__ void operator()() const override {
    printf("thread %d here from the special expert-kernel of 4 [val=%f]\n",
           threadIdx.x, val);
  }
  ValueT val = 0;
};

// Example for 1D generic
template <typename ValueT, int BLOCK_DIM_X, int BLOCK_DIM_Y>
struct ExpertKernel2D : public cuda::Kernel {
  void Launch(cudaStream_t stream = 0) override {
    dim3 block(BLOCK_DIM_X, BLOCK_DIM_Y);
    dim3 grid(1, 1);
    cuda::Run<<<grid, block, 0, stream>>>(*this);
  }

  __device__ __forceinline__ void operator()() const override {
    const int tid = threadIdx.x * BLOCK_DIM_Y + threadIdx.y;
    if (!tid)
      printf("thread %d here from the expert-kernel %d x %d [val=%f]\n", tid,
             BLOCK_DIM_X, BLOCK_DIM_Y, val);
  }
  ValueT val = 0;
};

// Example for 1D special
template <typename ValueT>
struct ExpertKernel2D<ValueT, 4, 3> : public cuda::Kernel {
  void Launch(cudaStream_t stream = 0) override {
    dim3 block(4, 3);
    dim3 grid(1, 1);
    cuda::Run<<<grid, block, 0, stream>>>(*this);
  }

  __device__ __forceinline__ void operator()() const override {
    const int tid = threadIdx.x * 3 + threadIdx.y;
    if (!tid)
      printf(
          "thread %d here from the special expert-kernel of 4 x 3 [val=%f]\n",
          tid, val);
  }
  ValueT val = 0;
};
}  // namespace

// Workaround to initialize all kernels for the dispatcher.
struct Initializer {
  explicit Initializer(float val) : val(val) {}

  template <typename TKernel>
  void operator()(TKernel* kernel) {
    kernel->val = val;
  }

  float val;
};

int main() {
  // We initialize these kernels using Initializer.
  // From c++14 on, we could use a lambda function.
  // But for now, we need this workaround.
  Initializer init(42.f);

  // Simple hyper-parameter:
  cuda::KernelDispatcher<int> disp(true);
  disp.Register<ExpertKernel1D<float, 4>>(3, init);
  disp.Register<ExpertKernel1D<float, 8>>(6, init);

  for (int i = 0; i < 9; ++i) {
    printf("%d : \n", i);
    disp.Run(i);
    ASSERT_CUDA(cudaDeviceSynchronize());
  }

  // Multi-dimensional hyper-parameters:
  cuda::KernelDispatcher<std::tuple<int, int>> disp2(true);
  disp2.Register<ExpertKernel2D<float, 4, 3>>(std::make_tuple(4, 3), init);
  disp2.Register<ExpertKernel2D<float, 8, 4>>(std::make_tuple(9, 4), init);

  for (int i = 0; i < 10; ++i) {
    for (int j = 0; j < 5; ++j) {
      printf("i: %d j %d\n", i, j);
      disp2.Run(std::make_tuple(i, j));
      ASSERT_CUDA(cudaDeviceSynchronize());
    }
  }

  return 0;
}

/*
0 :
thread 0 here from the special expert-kernel of 4 [val=42.000000]
thread 1 here from the special expert-kernel of 4 [val=42.000000]
thread 2 here from the special expert-kernel of 4 [val=42.000000]
thread 3 here from the special expert-kernel of 4 [val=42.000000]
1 :
thread 0 here from the special expert-kernel of 4 [val=42.000000]
thread 1 here from the special expert-kernel of 4 [val=42.000000]
thread 2 here from the special expert-kernel of 4 [val=42.000000]
thread 3 here from the special expert-kernel of 4 [val=42.000000]
2 :
thread 0 here from the special expert-kernel of 4 [val=42.000000]
thread 1 here from the special expert-kernel of 4 [val=42.000000]
thread 2 here from the special expert-kernel of 4 [val=42.000000]
thread 3 here from the special expert-kernel of 4 [val=42.000000]
3 :
thread 0 here from the special expert-kernel of 4 [val=42.000000]
thread 1 here from the special expert-kernel of 4 [val=42.000000]
thread 2 here from the special expert-kernel of 4 [val=42.000000]
thread 3 here from the special expert-kernel of 4 [val=42.000000]
4 :
thread 0 here from the expert-kernel 8 [val=42.000000]
thread 1 here from the expert-kernel 8 [val=42.000000]
thread 2 here from the expert-kernel 8 [val=42.000000]
thread 3 here from the expert-kernel 8 [val=42.000000]
thread 4 here from the expert-kernel 8 [val=42.000000]
thread 5 here from the expert-kernel 8 [val=42.000000]
thread 6 here from the expert-kernel 8 [val=42.000000]
thread 7 here from the expert-kernel 8 [val=42.000000]
5 :
thread 0 here from the expert-kernel 8 [val=42.000000]
thread 1 here from the expert-kernel 8 [val=42.000000]
thread 2 here from the expert-kernel 8 [val=42.000000]
thread 3 here from the expert-kernel 8 [val=42.000000]
thread 4 here from the expert-kernel 8 [val=42.000000]
thread 5 here from the expert-kernel 8 [val=42.000000]
thread 6 here from the expert-kernel 8 [val=42.000000]
thread 7 here from the expert-kernel 8 [val=42.000000]
6 :
thread 0 here from the expert-kernel 8 [val=42.000000]
thread 1 here from the expert-kernel 8 [val=42.000000]
thread 2 here from the expert-kernel 8 [val=42.000000]
thread 3 here from the expert-kernel 8 [val=42.000000]
thread 4 here from the expert-kernel 8 [val=42.000000]
thread 5 here from the expert-kernel 8 [val=42.000000]
thread 6 here from the expert-kernel 8 [val=42.000000]
thread 7 here from the expert-kernel 8 [val=42.000000]
7 :
thread 0 here from the expert-kernel 8 [val=42.000000]
thread 1 here from the expert-kernel 8 [val=42.000000]
thread 2 here from the expert-kernel 8 [val=42.000000]
thread 3 here from the expert-kernel 8 [val=42.000000]
thread 4 here from the expert-kernel 8 [val=42.000000]
thread 5 here from the expert-kernel 8 [val=42.000000]
thread 6 here from the expert-kernel 8 [val=42.000000]
thread 7 here from the expert-kernel 8 [val=42.000000]
8 :
thread 0 here from the expert-kernel 8 [val=42.000000]
thread 1 here from the expert-kernel 8 [val=42.000000]
thread 2 here from the expert-kernel 8 [val=42.000000]
thread 3 here from the expert-kernel 8 [val=42.000000]
thread 4 here from the expert-kernel 8 [val=42.000000]
thread 5 here from the expert-kernel 8 [val=42.000000]
thread 6 here from the expert-kernel 8 [val=42.000000]
thread 7 here from the expert-kernel 8 [val=42.000000]
i: 0 j 0
thread 0 here from the special expert-kernel of 4 x 3 [val=42.000000]
i: 0 j 1
thread 0 here from the special expert-kernel of 4 x 3 [val=42.000000]
i: 0 j 2
thread 0 here from the special expert-kernel of 4 x 3 [val=42.000000]
i: 0 j 3
thread 0 here from the special expert-kernel of 4 x 3 [val=42.000000]
i: 0 j 4
thread 0 here from the special expert-kernel of 4 x 3 [val=42.000000]
i: 1 j 0
thread 0 here from the special expert-kernel of 4 x 3 [val=42.000000]
i: 1 j 1
thread 0 here from the special expert-kernel of 4 x 3 [val=42.000000]
i: 1 j 2
thread 0 here from the special expert-kernel of 4 x 3 [val=42.000000]
i: 1 j 3
thread 0 here from the special expert-kernel of 4 x 3 [val=42.000000]
i: 1 j 4
thread 0 here from the special expert-kernel of 4 x 3 [val=42.000000]
i: 2 j 0
thread 0 here from the special expert-kernel of 4 x 3 [val=42.000000]
i: 2 j 1
thread 0 here from the special expert-kernel of 4 x 3 [val=42.000000]
i: 2 j 2
thread 0 here from the special expert-kernel of 4 x 3 [val=42.000000]
i: 2 j 3
thread 0 here from the special expert-kernel of 4 x 3 [val=42.000000]
i: 2 j 4
thread 0 here from the special expert-kernel of 4 x 3 [val=42.000000]
i: 3 j 0
thread 0 here from the special expert-kernel of 4 x 3 [val=42.000000]
i: 3 j 1
thread 0 here from the special expert-kernel of 4 x 3 [val=42.000000]
i: 3 j 2
thread 0 here from the special expert-kernel of 4 x 3 [val=42.000000]
i: 3 j 3
thread 0 here from the special expert-kernel of 4 x 3 [val=42.000000]
i: 3 j 4
thread 0 here from the special expert-kernel of 4 x 3 [val=42.000000]
i: 4 j 0
thread 0 here from the special expert-kernel of 4 x 3 [val=42.000000]
i: 4 j 1
thread 0 here from the special expert-kernel of 4 x 3 [val=42.000000]
i: 4 j 2
thread 0 here from the special expert-kernel of 4 x 3 [val=42.000000]
i: 4 j 3
thread 0 here from the special expert-kernel of 4 x 3 [val=42.000000]
i: 4 j 4
thread 0 here from the expert-kernel 8 x 4 [val=42.000000]
i: 5 j 0
thread 0 here from the expert-kernel 8 x 4 [val=42.000000]
i: 5 j 1
thread 0 here from the expert-kernel 8 x 4 [val=42.000000]
i: 5 j 2
thread 0 here from the expert-kernel 8 x 4 [val=42.000000]
i: 5 j 3
thread 0 here from the expert-kernel 8 x 4 [val=42.000000]
i: 5 j 4
thread 0 here from the expert-kernel 8 x 4 [val=42.000000]
i: 6 j 0
thread 0 here from the expert-kernel 8 x 4 [val=42.000000]
i: 6 j 1
thread 0 here from the expert-kernel 8 x 4 [val=42.000000]
i: 6 j 2
thread 0 here from the expert-kernel 8 x 4 [val=42.000000]
i: 6 j 3
thread 0 here from the expert-kernel 8 x 4 [val=42.000000]
i: 6 j 4
thread 0 here from the expert-kernel 8 x 4 [val=42.000000]
i: 7 j 0
thread 0 here from the expert-kernel 8 x 4 [val=42.000000]
i: 7 j 1
thread 0 here from the expert-kernel 8 x 4 [val=42.000000]
i: 7 j 2
thread 0 here from the expert-kernel 8 x 4 [val=42.000000]
i: 7 j 3
thread 0 here from the expert-kernel 8 x 4 [val=42.000000]
i: 7 j 4
thread 0 here from the expert-kernel 8 x 4 [val=42.000000]
i: 8 j 0
thread 0 here from the expert-kernel 8 x 4 [val=42.000000]
i: 8 j 1
thread 0 here from the expert-kernel 8 x 4 [val=42.000000]
i: 8 j 2
thread 0 here from the expert-kernel 8 x 4 [val=42.000000]
i: 8 j 3
thread 0 here from the expert-kernel 8 x 4 [val=42.000000]
i: 8 j 4
thread 0 here from the expert-kernel 8 x 4 [val=42.000000]
i: 9 j 0
thread 0 here from the expert-kernel 8 x 4 [val=42.000000]
i: 9 j 1
thread 0 here from the expert-kernel 8 x 4 [val=42.000000]
i: 9 j 2
thread 0 here from the expert-kernel 8 x 4 [val=42.000000]
i: 9 j 3
thread 0 here from the expert-kernel 8 x 4 [val=42.000000]
i: 9 j 4
thread 0 here from the expert-kernel 8 x 4 [val=42.000000]
 */
