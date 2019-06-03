OPTION(TEST_CUDA "Build Tests for CUDA" ON)
OPTION(SHOW_PTXAS "Build Tests for CUDA" ON)
set(CUDA_ARCH "" CACHE STRING "Target CUDA Architectures multiple are allowed")


# CUDA not available
if(CUDA_FOUND)

  # We can only build cuda tests if building cuda is enabled.
  message(STATUS "Build with CUDA support")

  if(TEST_CUDA)
    message(STATUS "Build tests for CUDA")
  endif(TEST_CUDA)


  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DWITH_CUDA ")
  include_directories(${CUDA_INCLUDE_DIRS})

  set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -std=c++11 --expt-relaxed-constexpr -DWITH_CUDA ")

  # Xptxas dumps register usage
  if(SHOW_PTXAS)
    set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS}  -Xptxas=-v")
  endif(SHOW_PTXAS)

  if(CMAKE_BUILD_TYPE STREQUAL "Release")
    message(STATUS "Build CUDA in ${CMAKE_BUILD_TYPE} mode")
    set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS}  -O3 -Ofast")
  endif(CMAKE_BUILD_TYPE STREQUAL "Release")

  if(CUDA_ARCH STREQUAL "")
    # good defaults for CUDA Toolkit 8.x
    if(CUDA_VERSION_MAJOR MATCHES 8)
      set(CUDA_ARCH "35 37 52 60")
    endif(CUDA_VERSION_MAJOR MATCHES 8)

    # good defaults for CUDA Toolkit 9.x
    if(CUDA_VERSION_MAJOR MATCHES 9)
      set(CUDA_ARCH "35 52 60 70")
    endif(CUDA_VERSION_MAJOR MATCHES 9)

    # good defaults for CUDA Toolkit 10.x
    if(CUDA_VERSION_MAJOR MATCHES 10)
      set(CUDA_ARCH "35 52 60 70")
    endif(CUDA_VERSION_MAJOR MATCHES 10)
  endif(CUDA_ARCH STREQUAL "")

  # str replace ' ' with ;
  STRING(REGEX REPLACE " " ";" CUDA_ARCH ${CUDA_ARCH})

  # set the compiler flags for each NV target
  foreach(target ${CUDA_ARCH})
    set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} -gencode=arch=compute_${target},code=\\\"sm_${target},compute_${target}\\\")
  endforeach(target ${CUDA_ARCH})

else(CUDA_FOUND)

  message(STATUS "Build CUDA and tests for CUDA are disabled")
  set(TEST_CUDA OFF)

endif(CUDA_FOUND)