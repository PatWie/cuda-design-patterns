CMAKE_MINIMUM_REQUIRED( VERSION 3.3.0 )
project(example)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_BUILD_TYPE Release)
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -Ofast -DNDEBUG -Wall -Wextra ")
enable_testing()

find_package(GTest REQUIRED)
include_directories(${GTEST_INCLUDE_DIRS})

OPTION(WITH_CUDA "Build functions with CUDA enabled" OFF)
OPTION(TEST_CUDA "Build Tests for CUDA" ON)


# We can only build cuda tests if building cuda is enabled.
if(WITH_CUDA)
  message(STATUS "Build with CUDA support")
  if(TEST_CUDA)
    message(STATUS "Build tests for CUDA")
  endif(TEST_CUDA)
else(WITH_CUDA)
  message(STATUS "Build CUDA is disabled")
  set(TEST_CUDA OFF)
endif(WITH_CUDA)


include_directories(
  ${CMAKE_CURRENT_SOURCE_DIR}/include
)

# add_library(multiply_cpu SHARED src/multiply/multiply_cpu.cc )
# list(APPEND LIBRARIES "multiply_cpu")

if(WITH_CUDA)
  list(APPEND CMAKE_CXX_FLAGS "-DWITH_CUDA")

  include_directories(
    ${CUDA_INCLUDE_DIRS}
  )

  find_package(CUDA REQUIRED)
  # Xptxas dumps register usage
  set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -std=c++11 -Xptxas=-v --expt-relaxed-constexpr -DWITH_CUDA")


  #CUDA Toolkit 8.x
  if(CUDA_VERSION_MAJOR MATCHES 8)
    set(CUDA_ARCH "35 37 52 60" CACHE STRING "Target Architectures (SM35 SM37 SM52 SM60), multiple are allowed")
  endif(CUDA_VERSION_MAJOR MATCHES 8)

  #CUDA Toolkit 9.x
  if(CUDA_VERSION_MAJOR MATCHES 9)
    set(CUDA_ARCH "35 52 60 70" CACHE STRING "Target Architectures (SM35 SM52 SM60 SM70), multiple are allowed")
  endif(CUDA_VERSION_MAJOR MATCHES 9)

  #CUDA Toolkit 10.x
  if(CUDA_VERSION_MAJOR MATCHES 10)
    set(CUDA_ARCH "35 52 60 70" CACHE STRING "Target Architectures (SM35 SM52 SM60 SM70), multiple are allowed")
  endif(CUDA_VERSION_MAJOR MATCHES 10)

  # str replace ' ' with ;
  STRING(REGEX REPLACE " " ";" CUDA_ARCH ${CUDA_ARCH})

  #set the compiler flags for each NV target
  foreach(target ${CUDA_ARCH})
    set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} -gencode=arch=compute_${target},code=\\\"sm_${target},compute_${target}\\\")
  endforeach(target ${CUDA_ARCH})
endif(WITH_CUDA)


# if(WITH_CUDA)
#   # prepare gpu functions
#   cuda_add_library(multiply_gpu SHARED src/multiply/multiply_gpu.cu )
#   list(APPEND LIBRARIES "multiply_gpu")
# endif(WITH_CUDA)

# # some example application
# add_executable(multiply src/multiply.cc )
# target_link_libraries(multiply LINK_PUBLIC ${LIBRARIES})

# # all CPU tests
# add_executable(test_cpu test/test_multiply.cc)
# target_link_libraries(test_cpu ${GTEST_LIBRARIES} multiply_cpu pthread)

# add_test(TestCpu test_cpu)

# if(TEST_CUDA)
#   cuda_add_executable(test_gpu test/test_multiply.cu)
#   target_link_libraries(test_gpu ${GTEST_LIBRARIES} multiply_gpu pthread)
#   add_test(TestGpu test_gpu)
# endif(TEST_CUDA)


cuda_add_executable(sharedmemory src/sharedmemory.cu )
cuda_add_executable(tune src/tune.cu )
