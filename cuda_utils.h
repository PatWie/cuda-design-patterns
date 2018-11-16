/* Copyright 2017 ComputerGraphics Tuebingen. All Rights Reserved.
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
    http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/
// Author: Patrick Wieschollek, <mail@patwie.com>, 2018

#ifndef LIB_CUDA_UTILS_H_
#define LIB_CUDA_UTILS_H_

#include <cuda_runtime.h>

namespace cuda_utils {

namespace impl {

template <size_t rank, size_t skip, size_t pos, size_t remaining>
struct pitch_helper {
  constexpr size_t call(const size_t dims_[rank]) const {
    return pitch_helper<rank, skip - 1, pos + 1, rank - pos - 1>().call(dims_);
  }
};

template <size_t rank, size_t pos, size_t remaining>
struct pitch_helper<rank, 0, pos, remaining> {
  constexpr size_t call(const size_t dims_[rank]) const {
    return dims_[pos] *
           pitch_helper<rank, 0, pos + 1, remaining - 1>().call(dims_);
  }
};

template <size_t rank, size_t pos>
struct pitch_helper<rank, 0, pos, 0> {
  constexpr size_t call(const size_t dims_[rank]) const { return 1; }
};

template <size_t rank, size_t remaining, class T, class... Ts>
struct position_helper {
  constexpr size_t call(const size_t dims_[rank], T v, Ts... is) const {
    return v * pitch_helper<rank, rank - remaining + 1, 0, rank>().call(dims_) +
           position_helper<rank, remaining - 1, Ts...>().call(dims_, is...);
  }
};

template <size_t rank, size_t remaining, class T>
struct position_helper<rank, remaining, T> {
  constexpr size_t call(const size_t dims_[rank], T v) const { return v; }
};

};  // namespace impl

template <size_t rank_>
struct BaseIndex {
 protected:
  size_t dims_[rank_];

 public:
  template <class... Ts>
  constexpr __device__ __forceinline__ BaseIndex(size_t i0, Ts... is) noexcept
      : dims_{i0, is...} {}

  template <class... Ts>
  constexpr __device__ __forceinline__ bool valid(size_t i0, Ts... is) const {
    return valid_impl<0, Ts...>(i0, is...);
  }

  constexpr __device__ __forceinline__ size_t rank() const { return rank_; }
  constexpr __device__ __forceinline__ size_t dim(size_t axis) const {
    return dims_[axis];
  }

 private:
  template <size_t num, class... Ts>
  constexpr __device__ __forceinline__ bool valid_impl(size_t i0,
                                                       Ts... is) const {
    return (i0 < dims_[num]) && valid_impl<num + 1, Ts...>(is...);
  }

  template <size_t num, typename T>
  constexpr __device__ __forceinline__ bool valid_impl(T i0) const {
    return (i0 < dims_[rank_ - 1]);
  }

 protected:
  template <class... Ts>
  constexpr __device__ __forceinline__ size_t _index(size_t i0,
                                                     Ts... is) const {
    return impl::position_helper<rank_, rank_, size_t, Ts...>().call(dims_, i0,
                                                                     is...);
  }
};

/**
 * Create an index object.
 *
 * The index object can handle various dimensions.
 *
 *     auto idx = Index<4>(B, H, W, C);
 *     auto pos = idx(b, h, w, c);
 *
 * @param rank in each dimensions.
 */
template <size_t rank_>
struct Index : public BaseIndex<rank_> {
 public:
  template <class... Ts>
  constexpr __device__ __forceinline__ Index(size_t i0, Ts... is) noexcept
      : BaseIndex<rank_>(i0, is...) {}

  template <class... Ts>
  size_t __device__ __forceinline__ operator()(size_t i0, Ts... is) const {
    return _index(i0, is...);
  }

  template <class... Ts>
  size_t __device__ __forceinline__ operator[](size_t i0) const {
    return BaseIndex<rank_>::dims_[i0];
  }
};

template <class T, size_t rank_>
struct Tensor_ : public BaseIndex<rank_> {
  T* data_;

 public:
  template <class... Ts>
  constexpr __device__ __forceinline__ Tensor_(T* data, size_t i0,
                                               Ts... is) noexcept
      : BaseIndex<rank_>(i0, is...), data_(data) {}

  /**
   * Returns value from given position if valid, else 0;
   *
   *    auto T = Tensor(data, A, B, C);
   *    auto val = T.safe_value(a, b, c);
   *
   * is equal
   *
   *    auto T = Tensor(data, A, B, C);
   *    auto val = T.valid(a, b, c) ? T(a, b, c) : 0;
   */
  template <class... Ts>
  T __device__ __forceinline__ safe_value(size_t i0, Ts... is) const {
    return valid(i0, is...) ? data_[index(i0, is...)] : 0;
  }

  /**
   * Returns value from given position if valid, else 0;
   *
   *    auto T = Tensor(data, A, B, C);
   *    auto val = T(a, b, c);
   */
  template <class... Ts>
  T __device__ __forceinline__ operator()(size_t i0, Ts... is) const {
    return data_[index(i0, is...)];
  }

  /**
   * Write value at given position.
   *
   *    auto T = Tensor(data, A, B, C);
   *    T(a, b, c) = 42;
   */
  template <class... Ts>
  T& __device__ operator()(size_t i0, Ts... is) {
    return data_[index(i0, is...)];
  }

  /**
   * Wrap c-array read access
   */
  template <class... Ts>
  T __device__ __forceinline__ operator[](size_t i0) const {
    return data_[i0];
  }

  /**
   * Wrap c-array write access
   */
  template <class... Ts>
  T& __device__ operator[](size_t i0) {
    return data_[i0];
  }

  /**
   * Returns index from given position.
   *    auto T = Tensor(data, A, B, C);
   *    size_t pos = T.index(a, b, c);
   */
  template <class TT, class... Ts>
  constexpr __device__ __forceinline__ TT index(TT i0, Ts... is) const {
    return _index(i0, is...);
  }

  T* flat() { return data_; }
};

/**
 * Create a tensor object.
 *
 * The tensor object is a combination of a flat array and nd-index.
 *
 *     const float* M = ...;
 *     auto Mt = Tensor(M, B, H, W, C);
 *     // same as auto Mt = Tensor<const float>(M, B, H, W, C);
 *     float val = Mt(b, h, w, c);
 *
 * WARNING, there is no double-check if the rank matches the number of
 * parameters.
 */
template <typename Dtype, class... Ts>
__device__ __forceinline__ auto Tensor(Dtype* arr, size_t N0, Ts... Ns)
    -> Tensor_<Dtype, size_t(1) + sizeof...(Ts)> {
  return Tensor_<Dtype, size_t(1) + sizeof...(Ts)>(arr, N0, Ns...);
}

/**
 * Create a tensor object but ensures rank.
 *
 * The tensor object is a combination of a flat array and nd-index.
 *
 *     const float* M = ...;
 *     auto Mt = Tensor<4>(M, B, H, W, C);
 *     // same as auto Mt = Tensor<const float>(M, B, H, W, C);
 *     float val = Mt(b, h, w, c);
 *
 * @param rank in each dimensions.
 */
template <size_t rank, typename Dtype, class... Ts>
__device__ __forceinline__ auto Tensor(Dtype* arr, size_t N0, Ts... Ns)
    -> Tensor_<Dtype, rank> {
  static_assert(size_t(1) + sizeof...(Ts) == rank,
                "Number of dimensions does not match rank! "
                "YOU_MADE_A_PROGAMMING_MISTAKE");
  return Tensor_<Dtype, rank>(arr, N0, Ns...);
}

/**
 * Proxy for shared memory when used in templates to avoid double extern.
 *
 *     run_kernel<<<grid, block, shm_size>>>(...)
 *
 *     Dtype* s_shm = DynamicSharedMemory<Dtype>();
 *     Dtype* s_el1 = (Dtype*)&s_shm[0];
 *     Dtype* s_el2 = (Dtype*)&s_shm[10];
 *
 * @param rank in each dimensions.
 */
template <typename T>
__device__ T* DynamicSharedMemory() {
  extern __shared__ __align__(sizeof(T)) unsigned char s_shm[];
  return reinterpret_cast<T*>(s_shm);
}

};      // namespace cuda_utils
#endif  // LIB_CUDA_UTILS_H_
