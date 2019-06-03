FROM nvidia/cuda:10.1-runtime

RUN apt-get update && apt-get install -y --no-install-recommends \
  cuda-libraries-dev-$CUDA_PKG_VERSION \
  cuda-nvml-dev-$CUDA_PKG_VERSION \
  cuda-minimal-build-$CUDA_PKG_VERSION \
  cuda-command-line-tools-$CUDA_PKG_VERSION \
  cmake \
  libnccl-dev=$NCCL_VERSION-1+cuda10.1 \
  xz-utils \
  build-essential \
  libgtest-dev \
  curl && \
  rm -rf /var/lib/apt/lists/*

RUN cd /usr/src/gtest && cmake CMakeLists.txt && make && cp *.a /usr/lib

ENV LIBRARY_PATH /usr/local/cuda/lib64/stubs
