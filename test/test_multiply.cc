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

#include "gtest/gtest.h"
#include "multiply/multiply.h"

namespace {

TEST(MultiplyTest, TestCpuIdentity) {
  float *A = new float[2 * 2];
  float *B = new float[2 * 2];
  float *expected = new float[2 * 2];

  for (int i = 0; i < 2 * 2; ++i) {
    A[i] = i;
    B[i] = 0;
    expected[i] = i;
  }
  B[0] = 1;
  B[3] = 1;

  float *actual = new float[2 * 2];

  Multiply<CPUDevice, float>::Apply(A, B, 2, 2, actual);

  for (int i = 0; i < 2 * 2; ++i) {
    EXPECT_EQ(expected[i], actual[i]);
  }
}

TEST(MultiplyTest, TestCpuSquare) {
  float *A = new float[3 * 3];
  float *B = new float[3 * 3];
  float *expected = new float[3 * 3];

  for (int i = 0; i < 3 * 3; ++i) {
    A[i] = i;
    B[i] = i;
  }
  expected[0] = 15;
  expected[1] = 18;
  expected[2] = 21;
  expected[3] = 42;
  expected[4] = 54;
  expected[5] = 66;
  expected[6] = 69;
  expected[7] = 90;
  expected[8] = 111;

  float *actual = new float[3 * 3];

  Multiply<CPUDevice, float>::Apply(A, B, 3, 3, actual);

  for (int i = 0; i < 3 * 3; ++i) {
    EXPECT_EQ(expected[i], actual[i]);
  }
}

}  // namespace

int main(int argc, char **argv) {
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
