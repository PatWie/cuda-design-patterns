# CUDA_UTILS

It is time consuming and not worth bothering with index calculations.
When writing CUDA code you usually have plenty of other important things to think about.

**I'm sick and tired of manually typing the indices.**
Further, each single character might introduce a bug.

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

Instead, this small library offers EIGEN-like index accessing:

```cpp
// Used 8 registers, 368 bytes cmem[0]
__global__ void readme_alternative(float *src, float *dst,
                                   int B, int H, int W, int C,
                                   int b, int h, int w, int c) {
  auto idx = Index<4>(B, H, W, c);
  dst[idx(b, h, w, c + 1)] = src[idx(b, h, w, c)];
}

// or even ...
// Used 8 registers, 368 bytes cmem[0]
__global__ void readme_alternative2(float *src, float *dst,
                                    int B, int H, int W, int C,
                                    int b, int h, int w, int c) {
  auto src_T = Tensor(src, B, H, W, C);
  auto dst_T = Tensor(dst, B, H, W, C);
  dst_T(b, h, w, c + 1) = src_T(b, h, w, c);
}
```

In such artificial examples, there is a small overhead.
For Matrix-Multiplication however, this library requires less registers:

```cpp
// Used 26 registers, 8192 bytes smem, 352 bytes cmem[0]
template <typename T, int num_threads>
__global__ void matrixMultiply_normal(T *C, const T *A,
                                      const T *B, int H,
                                      int W) {
  __shared__ T ds_M[num_threads][num_threads];
  __shared__ T ds_N[num_threads][num_threads];

  int tx = threadIdx.x;
  int ty = threadIdx.y;
  int Ch = blockIdx.y * num_threads + ty;
  int Cw = blockIdx.x * num_threads + tx;

  T Cval = 0;

  for (int m = 0; m < (W - 1) / num_threads + 1; ++m) {
    if (Ch < H && m * num_threads + tx < W)
      ds_M[ty][tx] = A[Ch * W + m * num_threads + tx];
    else
      ds_M[ty][tx] = 0;
    if (Cw < W && m * num_threads + ty < H)
      ds_N[ty][tx] = B[(m * num_threads + ty) * W + Cw];
    else
      ds_N[ty][tx] = 0;
    __syncthreads();

    for (int k = 0; k < num_threads; ++k) Cval += ds_M[ty][k] * ds_N[k][tx];
    __syncthreads();
  }
  if (Ch < H && Cw < W) C[Ch * W + Cw] = Cval;
}

// Used 22 registers, 8192 bytes smem, 352 bytes cmem[0]
template <typename T, int num_threads>
__global__ void matrixMultiply_alternative(T *C, const T *A,
                                           const T *B, int H,
                                           int W) {
  __shared__ T ds_M[num_threads][num_threads];
  __shared__ T ds_N[num_threads][num_threads];

  int tx = threadIdx.x;
  int ty = threadIdx.y;
  int Ch = blockIdx.y * num_threads + ty;
  int Cw = blockIdx.x * num_threads + tx;

  T Cval = 0;

  auto At = Tensor(A, H, W);    // supports both ways: without "<rank>"...
  auto Bt = Tensor<2>(B, H, W); // ... or with "<2>"
  auto Ct = Tensor<2>(C, H, W);

  for (int m = 0; m < (W - 1) / num_threads + 1; ++m) {
    ds_M[ty][tx] = At.safe_value(Ch, m * num_threads + tx);
    // ds_N[ty][tx] = Bt.safe_value(m * num_threads + ty, Cw);
    // is equivalent to:
    if (Bt.valid(m * num_threads + ty, Cw)) {
      ds_N[ty][tx] = Bt(m * num_threads + ty, Cw);
    }else{
      ds_N[ty][tx] = 0;
    }
    __syncthreads();

    for (int k = 0; k < num_threads; ++k) Cval += ds_M[ty][k] * ds_N[k][tx];
    __syncthreads();
  }
  if (Ct.valid(Ch, Cw)) Ct(Ch, Cw) = Cval;
}
```

There might be some tweaks to get identical ptx assembly.

# Benchmark

without `gencode`

|kernel|used registers (default) | used registers (this lib)|
|---|---|---|
| matmul | 26 | 22 |
| copy | 6 | 8 |
| index | 8 | 8 |
| flex-deconv | 42 | 48 (32) |
