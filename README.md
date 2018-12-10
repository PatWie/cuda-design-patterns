# CUDA_UTILS

It is time-consuming and not worthwhile to concern yourself with index calculations.
When writing CUDA code, you usually have many other vital things to ponder.

**I'm sick and tired of manually typing the indices.**
Each additional character increases the hit rate for a bug!

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
  auto idx = NdIndex<4>(B, H, W, c);
  dst[idx(b, h, w, c + 1)] = src[idx(b, h, w, c)];
}

// or even ...
// Used 8 registers, 368 bytes cmem[0]
__global__ void readme_alternative2(float *src, float *dst,
                                    int B, int H, int W, int C,
                                    int b, int h, int w, int c) {
  auto src_T = NdArray(src, B, H, W, C);
  auto dst_T = NdArray(dst, B, H, W, C);
  dst_T(b, h, w, c + 1) = src_T(b, h, w, c);
}

__global__ void readme_alternative2(NdArray<float, 4> src_T,
                                    NdArray<float, 4> dst_T,
                                    int b, int h, int w, int c) {
  dst_T(b, h, w, c + 1) = src_T(b, h, w, c);
}
```

There is a small overhead (cannot be avoided (?)) as we need to store each dimension of the array.
In some cases, the compiler can optimize this over-head "away". In some cases, such intermediate storage even
has benefits in terms of register usage.

Nonetheless, the code get much easier to read.

## Example

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

// Used 26 registers, 8192 bytes smem, 352 bytes cmem[0]
template <typename T, int num_threads>
__global__ void matrixMultiply_alternative(NdArray<float, 2> Ct,
                                           NdArray<const float, 2> At,
                                           NdArray<const float, 2> Bt) {
  __shared__ T ds_M[num_threads][num_threads];
  __shared__ T ds_N[num_threads][num_threads];

  const int tx = threadIdx.x;
  const int ty = threadIdx.y;
  const int Ch = blockIdx.y * num_threads + ty;
  const int Cw = blockIdx.x * num_threads + tx;
  const size_t W = Bt.dim(1);

  T Cval = 0;

  for (int m = 0; m < (W - 1) / num_threads + 1; ++m) {
    ds_M[ty][tx] = At.safe_value(Ch, m * num_threads + tx);
    ds_N[ty][tx] = Bt.safe_value(m * num_threads + ty, Cw);
    __syncthreads();

    for (int k = 0; k < num_threads; ++k) Cval += ds_M[ty][k] * ds_N[k][tx];
    __syncthreads();
  }
  if (Ct.valid(Ch, Cw)) Ct(Ch, Cw) = Cval;
}

// more lengthly version
// Used 22 registers, 8192 bytes smem, 352 bytes cmem[0]
template <typename T, int num_threads>
__global__ void matrixMultiply_alternative2(T *C, const T *A,
                                           const T *B, int H,
                                           int W) {
  __shared__ T ds_M[num_threads][num_threads];
  __shared__ T ds_N[num_threads][num_threads];

  int tx = threadIdx.x;
  int ty = threadIdx.y;
  int Ch = blockIdx.y * num_threads + ty;
  int Cw = blockIdx.x * num_threads + tx;

  T Cval = 0;

  auto At = make_ndarray<const T, 2>(A, H, W);
  auto Bt = make_ndarray<const T, 2>(B, H, W);
  auto Ct = make_ndarray<T, 2>(C, H, W);

  for (int m = 0; m < (W - 1) / num_threads + 1; ++m) {
    if (At.valid(Ch, m * num_threads + tx)) {
      ds_N[ty][tx] = At(Ch, m * num_threads + tx);
    }else{
      ds_N[ty][tx] = 0;
    }
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
