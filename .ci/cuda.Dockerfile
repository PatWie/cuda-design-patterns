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
  curl \
  unzip

RUN mkdir /google && cd /google && \
  curl https://github.com/google/googletest/archive/master.zip -O -J -L && \
  unzip googletest-master.zip  && \
  mv googletest-master src  && \
  rm googletest-master.zip  && \
  mkdir build && \
  mkdir dist && \
  cd build && \
  cmake ../src -DCMAKE_INSTALL_PREFIX=/google/dist && \
  make install

ENV GTEST_ROOT /google/dist
ENV LIBRARY_PATH /usr/local/cuda/lib64/stubs
