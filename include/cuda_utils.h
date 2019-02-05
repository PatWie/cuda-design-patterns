#ifndef LIB_CUDA_UTILS_H_
#define LIB_CUDA_UTILS_H_

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
 * Author: Patrick Wieschollek, <mail@patwie.com>, 2019
 *         Fabian Groh, <fabian.groh@uni-tuebingen.de>, 2019
 *
 */

#include <functional>
#include <map>
#include <memory>

// Template parameter for compile-time cuda drop-in replacements of cpu
// functions.
struct CPUDevice {
  static const bool cpu_device = true;
};

struct GPUDevice {
  static const bool cpu_device = false;
};

#if __CUDACC__
// __CUDACC__ is defined by nvcc on device and host
// __CUDA_ARCH__ is defined by nvcc on device

/**
 * This is the default way of testing whether executing the CUDA kernel has been
 * successfull.
 *
 * Example:
 *    Mykernel kernel;
 *    kernel.Launch();
 *    CHECK_CUDA(cudaDeviceSynchronize());
 *
 * @param  ans is a function that returns a cudaError_t
 * taken from: https://stackoverflow.com/a/14038590
 */
#define CHECK_CUDA(ans) \
  { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char* file, int line,
                      bool abort = true) {
  if (code != cudaSuccess) {
    fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file,
            line);
    if (abort) exit(code);
  }
}

namespace cuda {

/**
 * Compute the number of blocks for a given number of threads and a workload.
 * @param  N           number of workload instance
 * @param  num_threads number of threads per block
 * @return             number of required blocks
 */
__host__ __device__ __forceinline__ int divUp(int N, int num_threads) {
  return (N + num_threads - 1) / num_threads;
}

// Kernel is an abstract CUDA kernel, which can have attached values to avoid
// lengthly function signatures.
struct Kernel {
  /**
   * Launch contains the computation of all kernel parameters and executes
   * the CUDA call.
   *
   * This should include the computation of the kernel configuration like
   * gridDim, blockDim, shared_memory size. We enforce to use a cuda stream.
   *
   * Example:
   *   void Launch(cudaStream_t stream = 0){
   *     cuda::Run<<<1, 1, 0, stream>>>(*this);
   *   }
   *
   * @param stream used cuda stream
   */
  virtual void Launch(cudaStream_t stream = 0) = 0;

  /**
   * This operation contains the code, which will be executed on-chip.
   */
  virtual __device__ __forceinline__ void operator()() const = 0;
};

// Run a cuda kernel encapsulated in a struct.
// The kernel should have the following format
//
// struct Kernel {
//    void Launch(cudaStream_t stream = 0);
//    __device__ __forceinline__ void operator()();
// };
//
template <typename T>
__global__ void Run(const T kernel) {
  kernel();
}

/**
 * Benchmark will run a cuda kernel and benchmark the run-time of the kernel.
 *
 * Example:
 *    Mykernel kernel;
 *    printf("kernel took %f ms", cuda::Benchmark(&kernel));
 *
 * Note: This will execute a `cudaDeviceSynchronize` and should only be used for
 * debugging and benchmarking -- not in production!
 *
 * @param  kernel a struct containing the cuda kernel.
 * @return elapsed time in milli-seconds
 */
float Benchmark(Kernel* kernel) {
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  cudaEventRecord(start);
  kernel->Launch();
  cudaEventRecord(stop);

  // This stall the GPU execution pipeline and should not be used in production.
  CHECK_CUDA(cudaPeekAtLastError());
  CHECK_CUDA(cudaDeviceSynchronize());

  cudaEventSynchronize(stop);

  float milliseconds = 0;
  cudaEventElapsedTime(&milliseconds, start, stop);

  cudaEventDestroy(start);
  cudaEventDestroy(stop);

  return milliseconds;
}

/**
 * Proxy for shared memory when used in templates to avoid double extern.
 *
 *     run_kernel<<<grid, block, shm_size>>>(...)
 *
 *     T* s_shm = SharedMemory<T>();
 *     T* s_el1 = (T*)&s_shm[0];
 *     T* s_el2 = (T*)&s_shm[10]; // or use MixedSharedMemory
 *
 * @param rank in each dimensions.
 */
template <typename T>
__device__ T* SharedMemory() {
  extern __shared__ __align__(sizeof(T)) unsigned char s_shm[];
  return reinterpret_cast<T*>(s_shm);
}

/**
 * Extracting multiple values from shared memory of different types.
 *
 * Example:
 *    MixedSharedMemory shm;
 *    shm.add<float>(5);
 *    shm.add<int>(3);
 *    shm.add<float>(2);
 *
 *    kernel<<<...,...,shm.bytes>>>();
 *
 * and inside the CUDA kernel
 *
 *    MixedSharedMemory shm;
 *    float* shm_1 = shm.read<float>(5);
 *    int* shm_2 = shm.read<int>(3);
 *    float* shm_3 = shm.read<float>(2);
 */
struct MixedSharedMemory {
  int bytes = 0;
  char** shm_anchor;

  __host__ __device__ MixedSharedMemory() {
#if defined(__CUDA_ARCH__)
    extern __shared__ char* shm[];
    shm_anchor = shm;
#endif
  }

  template <typename T>
  __device__ T* read(int num) {
    T* ptr = reinterpret_cast<T*>(&shm_anchor[bytes]);
    bytes += num * sizeof(T);
    return ptr;
  }

  template <typename T>
  __host__ __device__ void add(int num) {
    bytes += num * sizeof(T);
  }
};

};  // namespace cuda

#endif  // __CUDACC__

namespace cuda {

struct EmptyInitializer {
  template <class T>
  void operator()(T* el) {}
};

/**
 * Dispatch template kernels according to a hyper parameter.
 *
 *   ExpertKernel<float, 4> kernelA;
 *   ExpertKernel<float, 8> kernelB;
 *   cuda::KernelDispatcher disp;
 *
 *   disp.Register<3>(kernelA); // for length up to 3 (inclusive) start kernelA
 *   disp.Register<6>(kernelB); // for length up to 6 (inclusive) start kernelB
 *
 *   int i = 6; // run-time variable
 *   disp.Run(i - 1); // launches kernelA
 *   disp.Run(i); // launches kernelB
 *   disp.Run(i + 1); // launches nothing
 */
template <typename KeyT = int, typename TComparator = std::less<KeyT>>
class KernelDispatcher {
  typedef std::function<void()> TLauncherFunc;
  typedef std::map<KeyT, TLauncherFunc, TComparator> TLauncherFuncMap;

 public:
  KernelDispatcher(bool extend = true) : extend(extend) {}

  // register
  template <typename T>
  void Register(KeyT bound, T& kernel) {
    Register(bound, [&]() { kernel.Launch(); });
  }

  // register and initialize
  template <typename T, typename Initializer>
  void Register(KeyT bound, T& kernel, Initializer = EmptyInitializer()) {
    Initializer()(&kernel);
    Register(bound, [&]() { kernel.Launch(); });
  }

  // would require C++14 to do
  // disp.Register(3, kernelA, [&](auto& T){T.val = 42;});
  //
  // template <typename T>
  // void Register(KeyT bound, T& kernel, std::function<void(T&)> init) {
  //   init(kernel);
  //   Register(bound, [&]() {
  //     kernel.Launch();
  //   });
  // }

  void Run(KeyT hyper) {
    typename TLauncherFuncMap::iterator detected_kernel =
        m_switchToVariant.lower_bound(hyper);
    if (detected_kernel == m_switchToVariant.end()) {
      if (extend) {
        m_switchToVariant.rbegin()->second();
      } else {
        // const KeyT upper_bound = m_switchToVariant.rbegin()->first;
        throw std::runtime_error(
            "KernelDispatcher has no kernels registered for the parameter "
            "requested by the runtime. Use 'KernelDispatcher(true)' to extend"
            " the range of the last registered kernel.");
      }
    } else {
      detected_kernel->second();
    }
  }

 private:
  void Register(KeyT bound, TLauncherFunc kernel) {
    m_switchToVariant[bound] = std::move(kernel);
  }

  TLauncherFuncMap m_switchToVariant;
  bool extend = true;
};
};  // namespace cuda

#endif  // LIB_CUDA_UTILS_H_
