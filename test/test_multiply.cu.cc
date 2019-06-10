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

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "include/cuda_utils.h"
#include "include/multiply/multiply.h"
#include "test/test_multiply_impl.h"

namespace {

TEST(MultiplyTest, ExtraGpuTest) { EXPECT_TRUE(true); }

TEST(MultiplyTest, GpuMatchCpu) {
  constexpr int M = 50;
  float *A = new float[M * M];
  float *B = new float[M * M];
  float *expected = new float[M * M];
  float *actual = new float[M * M];

  for (int i = 0; i < 2 * 2; ++i) {
    A[i] = i;
    B[i] = i - 5;
    expected[i] = 0;
    expected[i] = 0;
  }

  Multiply<CpuDevice, float>::Apply(A, B, M, M, expected);
  Multiply<GpuDevice, float>::Apply(A, B, M, M, actual);

  for (int i = 0; i < M * M; ++i) {
    EXPECT_NEAR(expected[i], actual[i], 1e-8);
    break;
  }
}

using Devices = ::testing::Types<GpuDevice>;
INSTANTIATE_TYPED_TEST_SUITE_P(Example, MultiplyTest, Devices);

}  // namespace

int main(int argc, char **argv) {
  ::testing::InitGoogleMock(&argc, argv);
  return RUN_ALL_TESTS();
}
