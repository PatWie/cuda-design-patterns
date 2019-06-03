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

#include <iostream>

#include "include/cuda_index.h"
#include "include/cuda_utils.h"

namespace {

struct AddSharedMemoryCUDAKernel : public cuda::Kernel {
  void Launch(cudaStream_t stream = 0) override {
    dim3 block(2);
    dim3 grid(1);

    cuda::SharedMemory shm;
    shm.add<float>(5);
    shm.add<int>(3);

    cuda::Run<<<grid, block, shm.bytes, stream>>>(*this);
  }

  __device__ __forceinline__ void operator()() const override {
    cuda::SharedMemory shm;
    float* floats_5 = shm.ref<float>(5);
    int* ints_3 = shm.ref<int>(3);

    if (threadIdx.x == 0) {
      floats_5[0] = 1.f;
      floats_5[1] = 2.f;
      floats_5[2] = 3.f;
      floats_5[3] = 4.f;
      floats_5[4] = 5.f;

      ints_3[0] = 11;
      ints_3[1] = 22;
      ints_3[2] = 33;
    }
    __syncthreads();
    if (threadIdx.x == 1) {
      float float_sum = 0;
      for (int i = 0; i < 5; ++i) {
        float_sum += floats_5[i];
        floats_5[i] = 0;
      }
      int int_sum = 0;
      for (int i = 0; i < 3; ++i) {
        int_sum += ints_3[i];
        ints_3[i] = 0;
      }

      printf("float sum: %f\n", float_sum);
      printf("int sum: %d\n", int_sum);
    }
  }
};
}  // namespace

int main(int argc, char const* argv[]) {
  AddSharedMemoryCUDAKernel kernel;
  kernel.Launch();
  ASSERT_CUDA(cudaDeviceSynchronize());
  return 0;
}

#endif  // __CUDACC__
