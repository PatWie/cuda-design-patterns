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

#include <stdio.h>

#include "include/cuda_utils.h"

void print_mat(float *A, int H, int W) {
  for (int h = 0; h < H; ++h) {
    for (int w = 0; w < W; ++w) {
      printf("%2.2f ", A[h * W + w]);
    }
    printf("\n");
  }
  printf("\n");
}

int main() {
  float *A = new float[2 * 2];
  float *B = new float[2 * 2];
  float *C = new float[2 * 2];
  for (int i = 0; i < 2 * 2; ++i) {
    A[i] = i;
    B[i] = i * 10;
    C[i] = 0;
  }

  print_mat(A, 2, 2);
  print_mat(B, 2, 2);

  // ...........................................................................
  printf("Cpu output\n");
  Multiply<CpuDevice, float>::Apply(A, B, 2, 2, C);
  print_mat(C, 2, 2);

// ...........................................................................
#if WITH_CUDA
  printf("Gpu output\n");
  for (int i = 0; i < 2 * 2; ++i) {
    C[i] = 0;
  }

  Multiply<GpuDevice, float>::Apply(A, B, 2, 2, C);

  print_mat(C, 2, 2);
#endif

  // ...........................................................................
  printf("auto output\n");
  for (int i = 0; i < 2 * 2; ++i) {
    C[i] = 0;
  }
  Multiply<XpuDevice, float>::Apply(A, B, 2, 2, C);

  print_mat(C, 2, 2);

  return 0;
}
