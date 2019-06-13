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
#include <map>
#include <memory>
#include <string>
#include <tuple>
#include <typeinfo>
#include <utility>

#ifdef __GNUG__
#include <cxxabi.h>
#include <cstdlib>
#endif
#include "include/cuda_utils.h"

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

namespace cuda {

/**
 * Dispatch template kernels according to a hyper parameter.
 *
 *   ExpertKernel<float, 4> kernelA;
 *   ExpertKernel<float, 8> kernelB;
 *   cuda::KernelBenchmarker disp(false);
 *
 *   disp.Register(3, kernelA); // for length up to 3 (inclusive) start kernelA
 *   disp.Register(6, kernelB); // for length up to 6 (inclusive) start kernelB
 *
 *   int i = 6;       // runtime variable
 *   disp.Run(i - 1); // launches kernelA
 *   disp.Run(i);     // launches kernelB
 *   disp.Run(i + 1); // triggers runtime exeception because of
 *                    // `disp(false)`
 */
template <typename KeyT = int, typename TComparator = std::less<KeyT>>
class KernelBenchmarker {
  using TLauncherFunc = std::function<void()>;
  using ValueT = std::tuple<TLauncherFunc, std::string>;
  using TLauncherFuncMap = std::map<KeyT, ValueT, TComparator>;

 public:
  // Taken from KernelDispatcher. Execuse the duplicated code. Templates cannot
  // have late bindings by virtual methods.
  // BEGIN DUPLICATE CODE

  // Register a instantiated kernel.
  //
  // Example
  //    cuda::KernelDispatcher<int> dispatcher;
  //    kernel<float, X> instance;
  //    dispatcher.Register(y, &instance);
  template <typename T>
  void Register(KeyT bound, T* kernel) {
    static_assert(impl::HasLaunchMethod<T>::value,
                  "The kernel struct needs to have a 'Launch()' method! "
                  "YOU_MADE_A_PROGAMMING_MISTAKE");
    Place<T>(bound, [&]() { kernel->Launch(); });
  }

  // Register and intialize a instantiated kernel.
  //
  // Example
  //    cuda::KernelDispatcher<int> dispatcher;
  //    kernel<float, X> instance;
  //    initializer init;
  //    dispatcher.Register(y, &instance, init);
  template <typename T, typename Initializer>
  void Register(KeyT bound, T* kernel, Initializer initializer) {
    static_assert(impl::HasLaunchMethod<T>::value,
                  "The kernel struct needs to have a 'Launch()' method! "
                  "YOU_MADE_A_PROGAMMING_MISTAKE");
    initializer(kernel);
    Place<T>(bound, [&]() { kernel->Launch(); });
  }

  // Register a kernel.
  //
  // Example
  //    cuda::KernelDispatcher<int> dispatcher;
  //    dispatcher.Register<kernel<float, X>>(y);
  template <typename T>
  void Register(KeyT bound) {
    static_assert(impl::HasLaunchMethod<T>::value,
                  "The kernel struct needs to have a 'Launch()' method! "
                  "YOU_MADE_A_PROGAMMING_MISTAKE");
    T kernel;
    Place<T>(bound, [&]() { kernel->Launch(); });
  }

  // Register and intialize a kernel.
  //
  // Example
  //    cuda::KernelDispatcher<int> dispatcher;
  //    initializer init;
  //    dispatcher.Register<kernel<float, X>>(y, init);
  template <typename T, typename Initializer>
  void Register(KeyT bound, Initializer initializer) {
    static_assert(impl::HasLaunchMethod<T>::value,
                  "The kernel struct needs to have a 'Launch()' method! "
                  "YOU_MADE_A_PROGAMMING_MISTAKE");
    T kernel;
    initializer(&kernel);
    Place<T>(bound, [&]() { kernel.Launch(); });
  }
  // END DUPLICATE CODE

  void Run() {
#if __CUDACC__
    std::cout << "Start Benchmark" << std::endl;

    for (auto&& kernel : kernels_) {
      const std::string name = std::get<1>(kernel.second);
      std::cout << "key " << kernel.first << " [" << name << "] ...";
      cudaEvent_t start, stop;
      cudaEventCreate(&start);
      cudaEventCreate(&stop);

      cudaEventRecord(start);
      std::get<0>(kernel.second)();
      cudaEventRecord(stop);

      ASSERT_CUDA(cudaPeekAtLastError());
      ASSERT_CUDA(cudaDeviceSynchronize());

      cudaEventSynchronize(stop);

      float milliseconds = 0;
      cudaEventElapsedTime(&milliseconds, start, stop);

      cudaEventDestroy(start);
      cudaEventDestroy(stop);

      std::cout << " took " << milliseconds << " ms" << std::endl;
    }

#endif  // __CUDACC__
  }

 private:
  template <typename T>
  void Place(KeyT bound, TLauncherFunc&& launch_func) {
    kernels_[bound] = std::make_tuple(std::forward<TLauncherFunc>(launch_func),
                                      demangle<0>(typeid(T).name()));
  }

  TLauncherFuncMap kernels_;
  bool extend = true;  // if true kernel with largest bound will act as default
};
}  // namespace cuda

#endif  // INCLUDE_CUDA_BENCHMARK_H_
