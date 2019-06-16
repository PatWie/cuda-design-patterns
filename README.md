# CUDA Design Patterns

| CUDA 9.0 | CUDA 9.1 | CUDA 10.0 | CUDA 10.1 |
| ------ | ------ | ------ | ------ |
| [![Build Status](https://ci.patwie.com/api/badges/PatWie/cuda-design-patterns/CUDA%209.0/status.svg)](https://ci.patwie.com/PatWie/cuda-design-patterns) | [![Build Status](https://ci.patwie.com/api/badges/PatWie/cuda-design-patterns/CUDA%209.1/status.svg)](https://ci.patwie.com/PatWie/cuda-design-patterns) | [![Build Status](https://ci.patwie.com/api/badges/PatWie/cuda-design-patterns/CUDA%2010.0/status.svg)](https://ci.patwie.com/PatWie/cuda-design-patterns) | [![Build Status](https://ci.patwie.com/api/badges/PatWie/cuda-design-patterns/CUDA%2010.1/status.svg)](https://ci.patwie.com/PatWie/cuda-design-patterns) |




Some best practises I collected over the last years when writing CUDA kernels. These functions
do not dictate how to use CUDA, these just simplify your workflow. I am not a big fan of libraries which rename things via wrappers. All code below does add additional benefits in CUDA programming.

## CUDA Boilerplate Code

[EXAMPLE](./src/multiply/multiply_gpu.cu.cc)

**Description:**
Avoid plain a CUDA kernel functions and instead pack them into a struct.


```cpp
template <typename ValueT>
struct MyKernel : public cuda::Kernel {
  void Launch(cudaStream_t stream = 0) {
    cuda::Run<<<1, 1, 0, stream>>>(*this);
  }
  __device__ __forceinline__ void operator()() const override {
    printf("hi from device code with value %f\n", val);
  }

  ValueT val;
};

MyKernel<float, 32> kernel;
kernel.val = 42.f;
kernel.Launch();
```

**Reasons:**

- This allows much better organization of used parameters. We recommend
to write them at the end of the struct, such that when writing the CUDA kernel itself
they are always visible.
- These structs can contain or compute the launch configuration (grid, block, shm size) depending on the parameters.
- Multiple kernel launches require less code, as we do not need to type out all parameters over and over again for a second or third launch.


## Functors

[EXAMPLE](./src/multiply.cc)

**Description:**
Use templated `structs` to switch seemlessly between CPU and GPU code:

```cpp
Multiply<float, CpuDevice>::Apply(A, B, 2, 2, C); // run CPU
Multiply<float, GpuDevice>::Apply(A, B, 2, 2, C); // run GPU
Multiply<float>::Apply(A, B, 2, 2, C); // run GPU if available else on CPU
```

**Reasons:**

- Switching between different devices is straight-forward.
- Understanding unit-tests which compare and verify the output becomes more easy.

## Shared Memory

[EXAMPLE](./src/sharedmemory.cu.cc)

Use

```cpp
cuda::SharedMemory shm;
float* floats_5 = shm.ref<float>(5);
int* ints_3 = shm.ref<int>(3);
```

instead of

```cpp
extern __shared__ char* shm[];
float* val1 = reinterpret_cast<float*>(&shm[0]); // 5 floats
int* val2 = reinterpret_cast<int*>(&shm[5]); // 3 ints
```


**Reasons:**

- The number of values of specific data types to read should be on the same line as the declaration. This way adding additional shared memory becomes easier during development.

## CUDA Kernel Dispatcher

[EXAMPLE](./src/tune.cu.cc)

Like in the *CUDA Boilerplate Code* example we pack our kernels into structs. For different hyper-parameters we use template specialization.

Given a generic CUDA kernel and a specialization

```cpp
template <typename ValueT, int BLOCK_DIM_X>
struct MyKernel : public cuda::Kernel {}

template <typename ValueT>
struct MyKernel<ValueT, 4> : public cuda::Kernel {}
```

we use the kernel dispatcher

```cpp
MyKernel<float, 4> kernelA;
MyKernel<float, 8> kernelB;

cuda::KernelDispatcher<int> dispatcher(true);
dispatcher.Register<MyKernel<float, 4>>(3); // for length up to 3 (inclusive) start MyKernel<float, 4>
dispatcher.Register<MyKernel<float, 8>>(6); // for length up to 6 (inclusive) start MyKernel<float, 8>
                                            // as `dispatcher(true)` this kernel will handle all
                                            // larger values as well
int i = 4;         // a runtime value
dispatcher.Run(i); // triggers `kernelB`
```

The dispatcher can also handle multi-dim values and a initializer

```cpp
struct Initializer {
  template <typename T>
  void operator()(T* el) {
    el->val = 42.f;
  }
};
Initializer init;
cuda::KernelDispatcher<std::tuple<int, int>> disp(true);
disp.Register<ExpertKernel2D<float, 4, 3>>(std::make_tuple(4, 3), init);
disp.Register<ExpertKernel2D<float, 8, 4>>(std::make_tuple(9, 4), init);
```

**Reasons:**

- Changing the block-dims will have performance impact. A templated CUDA kernel can execute special implementations for different hyper-parameters.
- A switch-statement dispatching run-time variables into a templated instantiation requires code-duplication, which can be avoid by the dispatcher.

## CUDA Index Calculation

[EXAMPLE](./src/deprecated_examples.cu_old)

Do not compute indicies by hand when appropriate and use

```cpp
// or even ...
// Used 8 registers, 368 bytes cmem[0]
__global__ void readme_alternative2(float *src, float *dst,
                                    int B, int H, int W, int C,
                                    int b, int h, int w, int c) {
  auto src_T = NdArray(src, B, H, W, C);
  auto dst_T = NdArray(dst, B, H, W, C);
  dst_T(b, h, w, c + 1) = src_T(b, h, w, c);
}
```

instead of

```cpp
// spot the bug
// Used 6 registers, 368 bytes cmem[0]
__global__ void readme_normal(float *src, float *dst,
                              int B, int H, int W, int C,
                              int b, int h, int w, int c) {
  const int pos1 = b * (H * W * C) + h * (W * c) + w * (C) + c;
  const int pos2 = b * (H * W * C) + h * (W * C) + w * (C) + (c + 1);
  dst[pos2] = src[pos1];
}
```

**Reasons**:

- It is time-consuming and not worthwhile to concern yourself with index calculations. When writing CUDA code, you usually have many other vital things to ponder.
- Each additional character increases the hit rate for a bug!
- **I'm sick and tired of manually typing the indices.**
- NdArray can have a positive impact on the number of used registers.

**Cons:**

- The compiler might not be able to optimize the `NdArray` overhead "away".
- NdArray can have a negative impact on the number of used registers.

## CMake Setup

**Description:**
Use CMake to configure which targets should be build. By default set `TEST_CUDA=ON` and `WITH_CUDA=OFF`.
The workflow (for this repository) is:

```bash
mkdir build && cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
# or more specific
cmake -DCMAKE_BUILD_TYPE=Release -DTEST_CUDA=ON -DCUDA_ARCH="52 60" ..
make
make test
```

**Reasons:**

-  Most CIs do not have a CUDA runtime installed. Whenever, `WITH_CUDA=ON` is activated the test code for CUDA will be also build.
-  FindCuda might be more robust than a custom makefile.

## Benchmark Kernels

[EXAMPLE](./src/benchmark-multiply.cu.cc)

**Description:**
Like in the *CUDA Boilerplate Code* example we pack our kernels into structs. We might want th benchmark different template arguments.

```cpp
cuda::KernelBenchmark<int> bench;
bench.Case<multiply_kernels::Multiply<float, 4>>(init);
bench.Case<multiply_kernels::Multiply<float, 6>>(init);
bench.Case<multiply_kernels::Multiply<float, 8>>(init);
bench.Case<multiply_kernels::Multiply<float, 16>>(init);
bench.Case<multiply_kernels::Multiply<float, 32>>(init);
bench.Start();
```

will give the output:

```
Using Device Number: 0
  Device name: GeForce GTX 970
  Memory Clock Rate (KHz): 3505000
  Memory Bus Width (bits): 256
  Peak Memory Bandwidth (GB/s): 224.320000

time 500.000000 - 1000.000000, iters: 5 - 100
 - multiply_kernels::Multiply<float, 4>    took     2.826743 ms stats(iters: 100, var:     0.067757, stddev:     0.260302)
 - multiply_kernels::Multiply<float, 6>    took     1.245100 ms stats(iters: 100, var:     0.019352, stddev:     0.139112)
 - multiply_kernels::Multiply<float, 8>    took     0.574468 ms stats(iters: 100, var:     0.000003, stddev:     0.001616)
 - multiply_kernels::Multiply<float, 16>   took     0.502195 ms stats(iters: 100, var:     0.000002, stddev:     0.001380)
 - multiply_kernels::Multiply<float, 32>   took     0.510635 ms stats(iters: 100, var:     0.000001, stddev:     0.001121)

```

## Tools
- [online CUDA calculator](http://cuda.patwie.com/) instead of the NVIDIA Excel-sheet
- [nvprof2json](https://github.com/PatWie/nvprof2json) to visualize NVIDIA profiling outputs in Google Chrome Browser (no dependencies compared to NVIDIA nvvp)