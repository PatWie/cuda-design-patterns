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

#ifndef INCLUDE_CUDA_BENCHMARK_H_
#define INCLUDE_CUDA_BENCHMARK_H_

#include <stdio.h>
#include <functional>
#include <iostream>
#include <memory>
#include <string>
#include <tuple>
#include <typeinfo>
#include <utility>
#include <vector>

#ifdef __GNUG__
#include <cxxabi.h>
#include <cstdlib>
#endif
#include "include/cuda_utils.h"

namespace cuda {

namespace impl {
// taken from https://stackoverflow.com/a/4541470/7443104
#ifdef __GNUG__

template <int DummyToBeInHeaderfile>
std::string demangle(const char* name) {
  int status = -4;  // some arbitrary value to eliminate the compiler warning

  // enable c++11 by passing the flag -std=c++11 to g++
  std::unique_ptr<char, void (*)(void*)> res{
      abi::__cxa_demangle(name, NULL, NULL, &status), std::free};

  return (status == 0) ? res.get() : name;
}

#else

// does nothing if not g++
template <int DummyToBeInHeaderfile>
std::string demangle(const char* name) {
  return name;
}

#endif
}  // namespace impl

/**
 * Benchmarks several templated kernels.
 *
 *   cuda::KernelBenchmarker<int> bench;
 *   bench.Case<multiply_kernels::Multiply<float, 2> >(init);
 *   bench.Case<multiply_kernels::Multiply<float, 4> >(init);
 *   bench.Case<multiply_kernels::Multiply<float, 8> >(init);
 *   bench.Case<multiply_kernels::Multiply<float, 16> >(init);
 *   bench.Case<multiply_kernels::Multiply<float, 32> >(init);
 *   bench.Run();
 */
template <typename KeyT = int>
class KernelBenchmarker {
  using TLauncherFunc = std::function<void()>;
  using ValueT = std::tuple<TLauncherFunc, std::string>;

 public:
  // Taken from KernelDispatcher. Execuse the duplicated code. Templates cannot
  // have late bindings by virtual methods.
  // BEGIN DUPLICATE CODE

  // Register a instantiated kernel.
  //
  // Example
  //    cuda::KernelDispatcher<int> dispatcher;
  //    kernel<float, X> instance;
  //    dispatcher.Case(&instance);
  template <typename T>
  void Case(T* kernel) {
    static_assert(impl::HasLaunchMethod<T>::value,
                  "The kernel struct needs to have a 'Launch()' method! "
                  "YOU_MADE_A_PROGAMMING_MISTAKE");
    Place<T>([&]() { kernel->Launch(); });
  }

  // Case and intialize a instantiated kernel.
  //
  // Example
  //    cuda::KernelDispatcher<int> dispatcher;
  //    kernel<float, X> instance;
  //    initializer init;
  //    dispatcher.Case(&instance, init);
  template <typename T, typename Initializer>
  void Case(T* kernel, Initializer initializer) {
    static_assert(impl::HasLaunchMethod<T>::value,
                  "The kernel struct needs to have a 'Launch()' method! "
                  "YOU_MADE_A_PROGAMMING_MISTAKE");
    initializer(kernel);
    Place<T>([&]() { kernel->Launch(); });
  }

  // Case a kernel.
  //
  // Example
  //    cuda::KernelDispatcher<int> dispatcher;
  //    dispatcher.Case<kernel<float, X>>();
  template <typename T>
  void Case() {
    static_assert(impl::HasLaunchMethod<T>::value,
                  "The kernel struct needs to have a 'Launch()' method! "
                  "YOU_MADE_A_PROGAMMING_MISTAKE");
    T* kernel = new T();  // neds to be on heap
    Place<T>([&]() { kernel->Launch(); });
  }

  // Case and intialize a kernel.
  //
  // Example
  //    cuda::KernelDispatcher<int> dispatcher;
  //    initializer init;
  //    dispatcher.Case<kernel<float, X>>(init);
  template <typename T, typename Initializer>
  void Case(Initializer initializer) {
    static_assert(impl::HasLaunchMethod<T>::value,
                  "The kernel struct needs to have a 'Launch()' method! "
                  "YOU_MADE_A_PROGAMMING_MISTAKE");
    T* kernel = new T();  // neds to be on heap
    initializer(kernel);
    Place<T>([&]() { kernel->Launch(); });
  }
  // END DUPLICATE CODE

  void Run() {
#if __CUDACC__
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    for (auto&& kernel : kernels_) {
      const std::string name = std::get<1>(kernel);
      std::cout << name << " ";

      cudaEventRecord(start);
      std::get<0>(kernel)();
      cudaEventRecord(stop);

      ASSERT_CUDA(cudaPeekAtLastError());
      ASSERT_CUDA(cudaDeviceSynchronize());

      cudaEventSynchronize(stop);

      float milliseconds = 0;
      cudaEventElapsedTime(&milliseconds, start, stop);

      std::cout << " took " << milliseconds << " ms" << std::endl;
    }

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

#endif  // __CUDACC__
  }

 private:
  template <typename T>
  void Place(TLauncherFunc&& launch_func) {
    kernels_.push_back(std::make_tuple(std::forward<TLauncherFunc>(launch_func),
                                       impl::demangle<0>(typeid(T).name())));
  }

  std::vector<ValueT> kernels_;
  bool extend = true;  // if true kernel with largest bound will act as default
};
}  // namespace cuda

#endif  // INCLUDE_CUDA_BENCHMARK_H_
