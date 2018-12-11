/* Copyright 2018 Authors. All Rights Reserved.

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
 * Author: Patrick Wieschollek, <mail@patwie.com>, 2018
 *
 */

#ifndef LIB_CUDA_UTILS_H_
#define LIB_CUDA_UTILS_H_

#define cuda_inline __device__ __host__ __forceinline__

#include <assert.h>
#include <cuda_runtime.h>

namespace cuda_utils {

namespace internal {

template <size_t TRank, size_t TSkip, size_t TPos, size_t TRemaining>
struct pitch_helper {
  constexpr size_t call(const size_t dimensions_[TRank]) const {
    return pitch_helper<TRank, TSkip - 1, TPos + 1, TRank - TPos - 1>().call(
        dimensions_);
  }
};

template <size_t TRank, size_t TPos, size_t TRemaining>
struct pitch_helper<TRank, 0, TPos, TRemaining> {
  constexpr size_t call(const size_t dimensions_[TRank]) const {
    return dimensions_[TPos] *
           pitch_helper<TRank, 0, TPos + 1, TRemaining - 1>().call(dimensions_);
  }
};

template <size_t TRank, size_t TPos>
struct pitch_helper<TRank, 0, TPos, 0> {
  constexpr size_t call(const size_t dimensions_[TRank]) const { return 1; }
};

template <size_t TRank, size_t TRemaining, class T, class... Ts>
struct position_helper {
  constexpr size_t call(const size_t dimensions_[TRank], T v, Ts... is) const {
    return v * pitch_helper<TRank, TRank - TRemaining + 1, 0, TRank>().call(
                   dimensions_) +
           position_helper<TRank, TRemaining - 1, Ts...>().call(dimensions_,
                                                                is...);
  }
};

template <size_t TRank, size_t TRemaining, class T>
struct position_helper<TRank, TRemaining, T> {
  constexpr size_t call(const size_t dimensions_[TRank], T v) const {
    return v;
  }
};

};  // namespace internal

template <size_t TRank>
struct BaseNdIndex {
 protected:
  size_t dimensions_[TRank];

 public:
  template <class... Ts>
  explicit constexpr cuda_inline BaseNdIndex(size_t i0, Ts... is) noexcept
      : dimensions_{i0, is...} {}

  /**
   * Check whether given coordinate is in range.
   */
  template <class... Ts>
  constexpr cuda_inline bool valid(size_t i0, Ts... is) const {
    return valid_impl<0, Ts...>(i0, is...);
  }

  /**
   * Return the number of axes.
   * @return number of axes
   */
  constexpr cuda_inline size_t rank() const { return TRank; }

  /**
   * Return the dimension for a given axis.
   *
   *     const size_t D = my_nd_array.template dim<1>();
   *
   * @return dimension for given axis
   */
  template <size_t TAxis>
  constexpr cuda_inline size_t dim() const {
    static_assert(TAxis < TRank, "axis < rank failed");
    return dimensions_[TAxis];
  }

 private:
  template <size_t TNum, class... Ts>
  constexpr cuda_inline bool valid_impl(size_t i0, Ts... is) const {
    return (i0 < dimensions_[TNum]) && valid_impl<TNum + 1, Ts...>(is...);
  }

  template <size_t TNum, typename T>
  constexpr cuda_inline bool valid_impl(T i0) const {
    return (i0 < dimensions_[TRank - 1]);
  }

 protected:
  template <class... Ts>
  constexpr cuda_inline size_t _index(size_t i0, Ts... is) const {
    return internal::position_helper<TRank, TRank, size_t, Ts...>().call(
        dimensions_, i0, is...);
  }
};

/**
 * Create an index object.
 *
 * The index object can handle various dimensions.
 *
 *     auto idx = NdIndex<4>(B, H, W, C);
 *     auto TPos = idx(b, h, w, c);
 *
 * @param rank in each dimensions.
 */
template <size_t TRank>
struct NdIndex : public BaseNdIndex<TRank> {
 public:
  template <class... Ts>
  explicit constexpr cuda_inline NdIndex(size_t i0, Ts... is) noexcept
      : BaseNdIndex<TRank>(i0, is...) {}

  /**
   * Get flattened index for a given position.
   *
   *     auto idx = NdIndex<4>(10, 20, 30, 40);
   *     size_t actual = idx(1, 2, 3, 4);
   *     size_t expected = 1 * (20 * 30 * 40) + 2 * (30 * 40) + 3 * (40) + 4;
   */
  template <class... Ts>
  size_t cuda_inline operator()(size_t i0, Ts... is) const {
    return _index(i0, is...);
  }

  /**
   * Get dimension for a given axis.
   *
   *     auto idx = NdIndex<4>(10, 20, 30, 40);
   *     size_t actual = idx[1]; // is 20
   */
  template <class... Ts>
  size_t cuda_inline operator[](size_t i0) const {
    return BaseNdIndex<TRank>::dimensions_[i0];
  }
};

template <class T, size_t TRank>
struct NdArray : public BaseNdIndex<TRank> {
  T* data_;

 public:
  template <class... Ts>
  explicit constexpr cuda_inline NdArray(T* data, size_t i0, Ts... is) noexcept
      : BaseNdIndex<TRank>(i0, is...), data_(data) {}

  /**
   * Returns value from given position if valid, else 0;
   *
   *    auto T = make_ndarray(data, A, B, C);
   *    auto val = T.safe_value(a, b, c);
   *
   * is equal
   *
   *    auto T = make_ndarray(data, A, B, C);
   *    auto val = T.valid(a, b, c) ? T(a, b, c) : 0;
   */
  template <class... Ts>
  T cuda_inline safe_value(size_t i0, Ts... is) const {
    return valid(i0, is...) ? data_[index(i0, is...)] : 0;
  }

  /**
   * Returns value from given position if valid, else 0;
   *
   *    auto T = make_ndarray(data, A, B, C);
   *    auto val = T(a, b, c);
   */
  template <class... Ts>
  T cuda_inline operator()(size_t i0, Ts... is) const {
    return data_[index(i0, is...)];
  }

  /**
   * Write value at given position.
   *
   *    auto T = make_ndarray(data, A, B, C);
   *    T(a, b, c) = 42;
   */
  template <class... Ts>
  T& __device__ __host__ operator()(size_t i0, Ts... is) {
    return data_[index(i0, is...)];
  }

  /**
   * Wrap c-array read access
   */
  template <class... Ts>
  T cuda_inline operator[](size_t i0) const {
    return data_[i0];
  }

  /**
   * Wrap c-array write access
   */
  template <class... Ts>
  T& __device__ __host__ operator[](size_t i0) {
    return data_[i0];
  }

  /**
   * Returns index from given position.
   *    auto T = make_ndarray(data, A, B, C);
   *    size_t TPos = T.index(a, b, c);
   */
  template <class TT, class... Ts>
  constexpr cuda_inline TT index(TT i0, Ts... is) const {
    return _index(i0, is...);
  }

  T* flat() { return data_; }
};

/**
 * Create a multi-dim. array object but ensures rank.
 *
 * The multi-dim. array object is a combination of a flat array and nd-index.
 *
 *     const float* M = ...;
 *     auto Mt = make_ndarray<float, 4>(M, B, H, W, C);
 *     float val = Mt(b, h, w, c);
 *
 * @param rank in each dimensions.
 */
template <typename T, size_t TRank, class... Ts>
cuda_inline auto make_ndarray(T* arr, size_t N0, Ts... Ns)
    -> NdArray<T, TRank> {
  static_assert(size_t(1) + sizeof...(Ts) == TRank,
                "Number of dimensions does not match rank! "
                "YOU_MADE_A_PROGAMMING_MISTAKE");
  return NdArray<T, TRank>(arr, N0, Ns...);
}

/**
 * Proxy for shared memory when used in templates to avoid double extern.
 *
 *     run_kernel<<<grid, block, shm_size>>>(...)
 *
 *     T* s_shm = DynamicSharedMemory<T>();
 *     T* s_el1 = (T*)&s_shm[0];
 *     T* s_el2 = (T*)&s_shm[10];
 *
 * @param rank in each dimensions.
 */
template <typename T>
__device__ __host__ T* DynamicSharedMemory() {
  extern __shared__ __align__(sizeof(T)) unsigned char s_shm[];
  return reinterpret_cast<T*>(s_shm);
}

};  // namespace cuda_utils

#undef cuda_inline
#endif  // LIB_CUDA_UTILS_H_
