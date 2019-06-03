# CUDA Design Patterns

[![Build Status](https://ci.patwie.com/api/badges/PatWie/cuda-design-patterns/status.svg)](https://ci.patwie.com/PatWie/cuda-design-patterns)

Some best practises I collected over the last years when writing CUDA kernels. These functions
do not dictate how to use CUDA, these just simplify your workflow. I am not a big fan of libraries which rename things via wrappers. All code below does add additional benefits in CUDA programming.

## CUDA Boilerplate Code

[EXAMPLE](./src/multiply/multiply_gpu.cu)

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
Multiply<CPUDevice, float>::Apply(A, B, 2, 2, C);
Multiply<GPUDevice, float>::Apply(A, B, 2, 2, C);
```

**Reasons:**

- Switching between different devices is straight-forward.
- Understanding unit-tests which compare and verify the output becomes more easy.

## Shared Memory

[EXAMPLE](./src/sharedmemory.cu)

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

[EXAMPLE](./src/tune.cu)

As in the *CUDA Boilerplate Code* example we pack our kernels into structs. For different hyper-parameters we use template specialization.

Given a generic CUDA kernel and a specialization

```cpp
template <typename ValueT, int BLOCK_DIM_X>
struct ExpertKernel : public cuda::Kernel {}

template <typename ValueT>
struct ExpertKernel<ValueT, 4> : public cuda::Kernel {}
```

we use the kernel dispatcher

```cpp
ExpertKernel<float, 4> kernelA;
ExpertKernel<float, 8> kernelB;

cuda::KernelDispatcher<int> disp(true);
disp.Register(3, kernelA); // for length up to 3 (inclusive) start kernelA
disp.Register(6, kernelB); // for length up to 6 (inclusive) start kernelB
                           // as `disp(true)` this kernel will handle all
                           // larger values as well
int i = 4; // a runtime value
disp.Run(i); // triggers `kernelB`
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
cuda2::KernelDispatcher<std::tuple<int, int>> disp2(true);
disp2.Register(std::make_tuple(4, 3), kernelA2d, init);
disp2.Register(std::make_tuple(9, 4), kernelB2d, init);
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

## Tools
- [online CUDA calculator](http://cuda.patwie.com/) instead of the NVIDIA Excel-sheet
- [nvprof2json](https://github.com/PatWie/nvprof2json) to visualize NVIDIA profiling outputs in Google Chrome Browser (no dependencies compared to NVIDIA nvvp)
