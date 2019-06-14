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
#include <iomanip>
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

namespace internal {
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
}  // namespace internal

/**
 * Benchmarks several templated kernels.
 *
 *   cuda::KernelBenchmark<int> bench;
 *   bench.Case<multiply_kernels::Multiply<float, 2>>(init);
 *   bench.Case<multiply_kernels::Multiply<float, 4>>(init);
 *   bench.Case<multiply_kernels::Multiply<float, 8>>(init);
 *   bench.Case<multiply_kernels::Multiply<float, 16>>(init);
 *   bench.Case<multiply_kernels::Multiply<float, 32>>(init);
 *   bench.Start();
 */
template <typename KeyT = int>
class KernelBenchmark {
  using TLauncherFunc = std::function<void()>;
  using ValueT = std::tuple<TLauncherFunc, std::string>;

  // we test at most 1 second
  const float max_time_ms = 1000;
  // we test at least 0.5 second
  const float min_time_ms = 500;
  // we test at most 100 times
  const int min_iterations = 5;
  const int max_iterations = 100;
  const int device_id = 0;

 public:
  // Register a kernel.
  //
  // Example
  //    cuda::KernelDispatcher<int> dispatcher;
  //    dispatcher.Case<kernel<float, X>>();
  template <typename T>
  void Case() {
    static_assert(internal::HasLaunchMethod<T>::value,
                  "The kernel struct needs to have a 'Launch()' method! "
                  "YOU_MADE_A_PROGAMMING_MISTAKE");
    // NOTE: std::shared_ptr<T>, std::unique_ptr<T> does not work here
    // eg. std::shared_ptr<T> kernel(new T());
    // so we delete these objects by collecting them
    T* kernel = new T();  // needs to be on heap
    deleter_.push_back([&]() { delete kernel; });
    Place<T>([&kernel]() { kernel->Launch(); });
  }

  // Register and intialize a kernel.
  //
  // Example
  //    cuda::KernelDispatcher<int> dispatcher;
  //    initializer init;
  //    dispatcher.Case<kernel<float, X>>(init);
  template <typename T, typename Initializer>
  void Case(Initializer initializer) {
    static_assert(internal::HasLaunchMethod<T>::value,
                  "The kernel struct needs to have a 'Launch()' method! "
                  "YOU_MADE_A_PROGAMMING_MISTAKE");
    // NOTE: std::shared_ptr<T>, std::unique_ptr<T> does not work here
    // eg. std::shared_ptr<T> kernel(new T());
    // so we delete these objects by collecting them
    T* kernel = new T();  // needs to be on heap
    deleter_.push_back([&]() { delete kernel; });
    initializer(kernel);
    Place<T>([&kernel]() { kernel->Launch(); });
  }

  KernelBenchmark() = default;
  KernelBenchmark(float min_time_ms, float max_time_ms, int min_iterations,
                  int max_iterations)
      : min_time_ms(min_time_ms),
        max_time_ms(max_time_ms),
        min_iterations(min_iterations),
        max_iterations(max_iterations) {}

  virtual ~KernelBenchmark() {
    for (auto del : deleter_) {
      del();
    }
  }

  void DeviceInfo() {
    ASSERT_CUDA(cudaSetDevice(device_id));
    cudaDeviceProp prop;
    ASSERT_CUDA(cudaGetDeviceProperties(&prop, device_id));
    printf("Using Device Number: %d\n", device_id);
    printf("  Device name: %s\n", prop.name);
    printf("  Memory Clock Rate (KHz): %d\n", prop.memoryClockRate);
    printf("  Memory Bus Width (bits): %d\n", prop.memoryBusWidth);
    printf("  Peak Memory Bandwidth (GB/s): %f\n\n",
           2.0 * prop.memoryClockRate * (prop.memoryBusWidth / 8) / 1.0e6);
  }

  void Start() {
#if __CUDACC__
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    int longest_name_len = 0;
    for (auto&& kernel : kernels_) {
      int len = std::get<1>(kernel).length();
      if (len > longest_name_len) {
        longest_name_len = len;
      }
    }
    DeviceInfo();
    printf("time %f - %f, iters: %d - %d\n", min_time_ms, max_time_ms,
           min_iterations, max_iterations);

    for (auto&& kernel : kernels_) {
      const std::string name = std::get<1>(kernel);
      printf(" - %-*s ", longest_name_len + 1, name.c_str());

      // burn in
      for (int i = 0; i < 5; ++i) {
        cudaEventRecord(start);
        std::get<0>(kernel)();
        cudaEventRecord(stop);
      }

      // real measurement
      float total_milliseconds = 0;
      int used_iterations = 0;

      float old_mean_time = 0;
      float cur_mean_time = 0;
      float old_var_time = 0.0;
      float cur_var_time = 0.0;

      for (int counter = 0; counter < max_iterations;
           counter++, used_iterations++) {
        // measure kernel
        float milliseconds = 0;
        cudaEventRecord(start);
        std::get<0>(kernel)();
        cudaEventRecord(stop);
        ASSERT_CUDA(cudaPeekAtLastError());
        ASSERT_CUDA(cudaDeviceSynchronize());
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&milliseconds, start, stop);
        total_milliseconds += milliseconds;

        // Estimate if the result is stable enough to be reported.
        // We want to run at least two runs (variance needs this).
        if (counter > 0) {
          // Update running statistics.
          cur_mean_time =
              old_mean_time + (milliseconds - old_mean_time) / (counter + 1);
          cur_var_time = old_var_time + (milliseconds - old_mean_time) *
                                            (milliseconds - cur_mean_time);

          old_var_time = cur_var_time;
          old_mean_time = cur_mean_time;

          if (total_milliseconds <= min_time_ms) {
            continue;
          }

          // We can stop if it took already too long.
          if (total_milliseconds > max_time_ms) {
            break;
          }

          // We want at least some iterations.
          if (counter >= min_iterations) {
            // Is std-dev small enough?
            float real_stdev = sqrt(cur_var_time / (used_iterations - 1));
            if (real_stdev < 0.01 * cur_mean_time) {
              break;
            }
          }

        } else {
          old_mean_time = milliseconds;
          cur_mean_time = milliseconds;
        }
      }

      printf(" took %12f ms stats(iters: %3d, var: %12f, stddev: %12f)\n",
             total_milliseconds / used_iterations, used_iterations,
             cur_var_time / (used_iterations - 1),
             sqrt(cur_var_time / (used_iterations - 1)));
    }
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

#endif  // __CUDACC__
  }

 private:
  template <typename T>
  void Place(TLauncherFunc&& launch_func) {
    kernels_.push_back(
        std::move(std::make_tuple(std::forward<TLauncherFunc>(launch_func),
                                  internal::demangle<0>(typeid(T).name()))));
  }

  std::vector<TLauncherFunc> deleter_;
  std::vector<ValueT> kernels_;
  bool extend = true;  // if true kernel with largest bound will act as default
};
}  // namespace cuda

#endif  // INCLUDE_CUDA_BENCHMARK_H_
