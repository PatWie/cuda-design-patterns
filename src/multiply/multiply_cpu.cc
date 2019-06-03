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

#include "include/multiply/multiply.h"

#include "include/cuda_utils.h"

template <typename ValueT>
struct Multiply<CPUDevice, ValueT> {
  static void Apply(const ValueT* A, const ValueT* B, const int H, const int W,
                    ValueT* C) {
    for (int h = 0; h < H; ++h) {
      for (int w = 0; w < W; ++w) {
        C[h * W + w] = 0;
        for (int k = 0; k < W; ++k) {
          C[h * W + w] += A[h * W + k] * B[k * H + w];
        }
      }
    }
  }
};

template struct Multiply<CPUDevice, double>;
template struct Multiply<CPUDevice, float>;
template struct Multiply<CPUDevice, int>;
