#include <stdlib.h>
#include <algorithm>
#include <iostream>
#include "cuda_utils.h"

/*
nvcc examples.cu --expt-relaxed-constexpr -Xptxas="-v" -std=c++11 -o test
*/

////////////////////////////////////////////////////////////////////////////////

using cuda_utils::make_ndarray;
using cuda_utils::NdArray;
using cuda_utils::NdIndex;

#define check_cuda_call(ans) \
  { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line,
                      bool abort = true) {
  if (code != cudaSuccess) {
    fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file,
            line);
    if (abort) exit(code);
  }
}

template <typename T, int num_threads>
__global__ void matrixMultiply____________normal__________(T *C, const T *A,
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

template <typename T, int num_threads>
__global__ void matrixMultiply____________tensor__________(T *C, const T *A,
                                                           const T *B, int H,
                                                           int W) {
  __shared__ T ds_M[num_threads][num_threads];
  __shared__ T ds_N[num_threads][num_threads];

  const int tx = threadIdx.x;
  const int ty = threadIdx.y;
  const int Ch = blockIdx.y * num_threads + ty;
  const int Cw = blockIdx.x * num_threads + tx;

  T Cval = 0;

  auto At = make_ndarray<const T, 2>(A, H, W);
  auto Bt = make_ndarray<const T, 2>(B, H, W);
  auto Ct = make_ndarray<T, 2>(C, H, W);

  for (int m = 0; m < (W - 1) / num_threads + 1; ++m) {
    ds_M[ty][tx] = At.safe_value(Ch, m * num_threads + tx);
    // ds_N[ty][tx] = Bt.safe_value(m * num_threads + ty, Cw);
    if (Bt.valid(m * num_threads + ty, Cw)) {
      ds_N[ty][tx] = Bt(m * num_threads + ty, Cw);
    } else {
      ds_N[ty][tx] = 0;
    }
    __syncthreads();

    for (int k = 0; k < num_threads; ++k) Cval += ds_M[ty][k] * ds_N[k][tx];
    __syncthreads();
  }
  if (Ct.valid(Ch, Cw)) Ct(Ch, Cw) = Cval;
}

template <typename T, int num_threads>
__global__ void matrixMultiply____________tensor2__________(
    NdArray<T, 2> Ct, NdArray<const T, 2> At, NdArray<const T, 2> Bt) {
  __shared__ T ds_M[num_threads][num_threads];
  __shared__ T ds_N[num_threads][num_threads];

  const int tx = threadIdx.x;
  const int ty = threadIdx.y;
  const int Ch = blockIdx.y * num_threads + ty;
  const int Cw = blockIdx.x * num_threads + tx;
  const size_t W = Bt.template dim<1>();

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

/************* INDEX SIMPLE ***************************************************/

__global__ void index____________normal__________(int A, int B, int C, int a,
                                                  int b, int c) {
  const int idx = a * (B * C) + b * C + c;
  printf("value is %i\n", idx);
}

__global__ void index____________tensor__________(int A, int B, int C, int a,
                                                  int b, int c) {
  auto idx = NdIndex<3>(A, B, C);
  printf("value is %i\n", idx(a, b, c));
}

template <typename T>
__device__ __forceinline__ const T NAIVE_IDX(const T A, const T B, const T C,
                                             T a, T b, T c) {
  return a * B * C + b * C + c;
}

__global__ void index____________naive__________(int A, int B, int C, int a,
                                                 int b, int c) {
  const int idx = NAIVE_IDX(A, B, C, a, b, c);
  printf("value is %i\n", idx);
}

/************* README EXAMPLE *************************************************/

__global__ void readme____________normal__________(float *src, float *dst,
                                                   int B, int H, int W, int C,
                                                   int b, int h, int w, int c) {
  const int pos1 = b * (H * W * C) + h * (W * C) + w * (C) + c;
  const int pos2 = b * (H * W * C) + h * (W * C) + w * (C) + (c + 1);
  dst[pos2] = src[pos1];
}

__global__ void readme____________tensor__________(float *src, float *dst,
                                                   int B, int H, int W, int C,
                                                   int b, int h, int w, int c) {
  auto idx = NdIndex<4>(B, H, W, C);
  src[idx(b, h, w, c)] = dst[idx(b, h, w, c)];
  // auto src_t = Tensor(src, B, H, W, C);
  // auto dst_t = Tensor(dst, B, H, W, C);
  // src_t(b, h, w, c) = dst_t(b, h, w, c);
}
/************* FLEX-DECONV ***************************************************/
// Used 42 registers, 392 bytes cmem[0]
// taken from https://github.com/cgtuebingen/Flex-Convolution
template <typename Dtype>
__global__ void flex_deconv_simple(const int B, const int N, const int K,
                                   const int Dp, const int Din, const int Dout,
                                   const Dtype *positions,
                                   const Dtype *features,
                                   const int *neighborhood, const Dtype *theta,
                                   const Dtype *bias, Dtype *output) {
  const int b = blockIdx.z;

  for (int n = blockIdx.y * blockDim.y + threadIdx.y; n < N;
       n += blockDim.y * gridDim.y) {
    const int self_k = neighborhood[b * K * N + 0 * N + n];

    for (int k_ = 0; k_ < K; ++k_) {
      const int other_k = neighborhood[b * K * N + k_ * N + n];

      for (int dout = blockIdx.x * blockDim.x + threadIdx.x; dout < Dout;
           dout += blockDim.x * gridDim.x) {
        for (int din = 0; din < Din; ++din) {
          const Dtype v = features[b * Din * N + din * N + self_k];
          Dtype W = bias[din * Dout + dout];

          for (int dp = 0; dp < Dp; ++dp) {
            Dtype delta = positions[b * Dp * N + dp * N + other_k] -
                          positions[b * Dp * N + dp * N + self_k];
            W += theta[dp * Din * Dout + din * Dout + dout] * delta;
          }

          Dtype Wv = W * v;
          // this has been an atomic add
          output[b * Dout * N + dout * N + other_k] += Wv;
        }
      }
    }
  }
}

// Used 48 registers, 392 bytes cmem[0]
template <typename T>
__global__ void flex_deconv_tensor(const int B, const int N, const int K,
                                   const int Dp, const int Din, const int Dout,
                                   const T *positions, const T *features,
                                   const int *neighborhood, const T *theta,
                                   const T *bias, T *output) {
  auto pos_t = make_ndarray<const T, 3>(positions, B, Dp, N);
  auto feat_t = make_ndarray<const T, 3>(features, B, Din, N);
  auto theta_t = make_ndarray<const T, 3>(theta, Dp, Din, Dout);
  auto bias_t = make_ndarray<const T, 2>(bias, Din, Dout);
  auto neighborhood_t = make_ndarray<const int, 3>(neighborhood, B, K, N);
  auto output_t = make_ndarray<T, 3>(output, B, Dout, N);

  const int b = blockIdx.z;

  for (int n = blockIdx.y * blockDim.y + threadIdx.y; n < N;
       n += blockDim.y * gridDim.y) {
    const int self_k = neighborhood_t(b, 0, n);

    for (int k_ = 0; k_ < K; ++k_) {
      const int other_k = neighborhood_t(b, k_, n);

      for (int dout = blockIdx.x * blockDim.x + threadIdx.x; dout < Dout;
           dout += blockDim.x * gridDim.x) {
        for (int din = 0; din < Din; ++din) {
          const T v = feat_t(b, din, self_k);
          T W = bias_t(din, dout);

          for (int dp = 0; dp < Dp; ++dp) {
            T delta = pos_t(b, dp, other_k) - pos_t(b, dp, self_k);
            W += theta_t(dp, din, dout) * delta;
          }

          T Wv = W * v;
          output_t(b, dout, other_k) += Wv;
        }
      }
    }
  }
}

int up2(int len, int th) { return (len - 1) / th + 1; }
void run_flex_deconv() {
  // this will fail, but during compilation, we will see register usage
  int B = 8;
  int N = 1024;
  int K = 8;
  int Dp = 3;
  int Din = 64;
  int Dout = 64;

  float *positions_;
  float *features_;
  int *neighborhood_;
  float *theta_;
  float *bias_;
  float *output_;

  const int threads = 32;
  dim3 block(threads, threads, 1);
  dim3 grid(up2(Dout, threads), up2(N, threads), B);

  flex_deconv_simple<float><<<grid, block>>>(B, N, K, Dp, Din, Dout, positions_,
                                             features_, neighborhood_, theta_,
                                             bias_, output_);
  flex_deconv_tensor<float><<<grid, block>>>(B, N, K, Dp, Din, Dout, positions_,
                                             features_, neighborhood_, theta_,
                                             bias_, output_);
}

void run_readme() {
  int B = 4;
  int H = 17;
  int W = 32;
  int C = 32;
  float *d_src;
  float *d_dst;
  check_cuda_call(cudaMalloc(&d_src, sizeof(float) * B * H * W * C));
  check_cuda_call(cudaMalloc(&d_dst, sizeof(float) * B * H * W * C));

  int b = 1;
  int h = 3;
  int w = 3;
  int c = 8;
  dim3 grid1(1);
  dim3 block1(1);
  readme____________normal__________<<<grid1, block1>>>(d_src, d_dst, B, H, W,
                                                        C, b, h, w, c);
  readme____________tensor__________<<<grid1, block1>>>(d_src, d_dst, B, H, W,
                                                        C, b, h, w, c);
}

void run_simple() {
  int A = 4;
  int B = 17;
  int C = 32;

  int a = 1;
  int b = 3;
  int c = 8;
  dim3 grid1(1);
  dim3 block1(1);
  index____________normal__________<<<grid1, block1>>>(A, B, C, a, b, c);
  index____________tensor__________<<<grid1, block1>>>(A, B, C, a, b, c);
  index____________naive__________<<<grid1, block1>>>(A, B, C, a, b, c);
}

void run_matmul() {
  int H = 4;
  int W = 5;
  float *matA = new float[H * W];
  float *matB = new float[H * W];
  float *matC1 = new float[H * W];
  float *matC2 = new float[H * W];
  float *matC3 = new float[H * W];

  for (int i = 0; i < H * W; ++i) {
    matA[i] = rand() / static_cast<float>(RAND_MAX);
    matB[i] = rand() / static_cast<float>(RAND_MAX);
    matC1[i] = rand() / static_cast<float>(RAND_MAX);
    matC2[i] = rand() / static_cast<float>(RAND_MAX);
    matC3[i] = rand() / static_cast<float>(RAND_MAX);
  }

  float *d_matA;
  float *d_matB;
  float *d_matC1;
  float *d_matC2;
  float *d_matC3;

  check_cuda_call(cudaMalloc(&d_matA, sizeof(float) * H * W));
  check_cuda_call(cudaMalloc(&d_matB, sizeof(float) * H * W));
  check_cuda_call(cudaMalloc(&d_matC1, sizeof(float) * H * W));
  check_cuda_call(cudaMalloc(&d_matC2, sizeof(float) * H * W));
  check_cuda_call(cudaMalloc(&d_matC3, sizeof(float) * H * W));

  check_cuda_call(
      cudaMemcpy(d_matA, matA, sizeof(float) * H * W, cudaMemcpyHostToDevice));
  check_cuda_call(
      cudaMemcpy(d_matB, matB, sizeof(float) * H * W, cudaMemcpyHostToDevice));
  check_cuda_call(cudaMemcpy(d_matC1, matC1, sizeof(float) * H * W,
                             cudaMemcpyHostToDevice));
  check_cuda_call(cudaMemcpy(d_matC2, matC2, sizeof(float) * H * W,
                             cudaMemcpyHostToDevice));
  check_cuda_call(cudaMemcpy(d_matC3, matC3, sizeof(float) * H * W,
                             cudaMemcpyHostToDevice));

  const int num_threads = 32;
  dim3 threads(num_threads, num_threads);
  dim3 grid((W + 1) / num_threads + 1, (W + 1) / num_threads + 1);

  matrixMultiply____________normal__________<float, 32>
      <<<grid, threads>>>(d_matC1, d_matA, d_matB, H, W);

  check_cuda_call(cudaPeekAtLastError());
  check_cuda_call(cudaGetLastError());
  check_cuda_call(cudaDeviceSynchronize());

  matrixMultiply____________tensor__________<float, 32>
      <<<grid, threads>>>(d_matC2, d_matA, d_matB, H, W);

  check_cuda_call(cudaPeekAtLastError());
  check_cuda_call(cudaGetLastError());
  check_cuda_call(cudaDeviceSynchronize());

  auto Ct = make_ndarray<float, 2>(d_matC3, H, W);
  auto At = make_ndarray<const float, 2>(d_matA, H, W);
  auto Bt = make_ndarray<const float, 2>(d_matB, H, W);

  matrixMultiply____________tensor2__________<float, 32>
      <<<grid, threads>>>(Ct, At, Bt);

  check_cuda_call(cudaPeekAtLastError());
  check_cuda_call(cudaGetLastError());
  check_cuda_call(cudaDeviceSynchronize());

  check_cuda_call(cudaMemcpy(matC1, d_matC1, H * W * sizeof(float),
                             cudaMemcpyDeviceToHost));
  check_cuda_call(cudaMemcpy(matC2, d_matC2, H * W * sizeof(float),
                             cudaMemcpyDeviceToHost));
  check_cuda_call(cudaMemcpy(matC3, d_matC3, H * W * sizeof(float),
                             cudaMemcpyDeviceToHost));

  // verify
  bool good = true;
  printf("\n");
  for (int i = 0; i < H * W; ++i) {
    if (fabs(matC1[i] - matC2[i]) > 1e-8) {
      printf("%i %f %f %f ", i, matC1[i], matC2[i], matA[i]);
      good = false;
    }
    if (fabs(matC1[i] - matC3[i]) > 1e-8) {
      printf("%i %f %f %f ", i, matC1[i], matC3[i], matA[i]);
      good = false;
    }
  }
  printf("\n");
  if (good)
    printf("good\n");
  else
    printf("bad\n");
}

/******************************************************************************/

int main() {
  run_matmul();
  // run_readme();
  // run_simple();
  // run_flex_deconv();
  return 0;
}
