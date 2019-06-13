CMAKE_MINIMUM_REQUIRED( VERSION 2.8 )
project(example)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -Ofast -Wall -Wextra ")
enable_testing()

list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)

find_package(GTest REQUIRED)
include_directories(${GTEST_INCLUDE_DIRS})

find_package(CUDA)
include(CudaHelper)

include_directories(${CMAKE_CURRENT_SOURCE_DIR})

add_library(multiply_cpu SHARED src/multiply/multiply_cpu.cc src/multiply/multiply.cc )
list(APPEND LIBRARIES "multiply_cpu")

if(CUDA_FOUND)
  cuda_add_library(multiply_gpu SHARED src/multiply/multiply_gpu.cu.cc src/multiply/multiply.cc )
  list(APPEND LIBRARIES "multiply_gpu")

  cuda_add_executable(sharedmemory src/sharedmemory.cu.cc )
  cuda_add_executable(tune src/tune.cu.cc )
endif(CUDA_FOUND)

add_executable(multiply src/multiply.cc )
target_link_libraries(multiply LINK_PUBLIC ${LIBRARIES})

# Benchmark
if(CUDA_FOUND)
  cuda_add_executable(benchmark src/benchmark-multiply.cu.cc src/multiply/multiply_gpu.cu.cc src/multiply/multiply.cc)
endif(CUDA_FOUND)

# TESTS
# TODO(): find a more robust way to link gmock, see docker file for setup
# This currently assumes gmock.a is next to gtest.a
link_directories($ENV{GTEST_ROOT}/lib)
add_executable(test_cpu test/test_multiply.cc)
target_link_libraries(test_cpu ${GTEST_LIBRARIES} gmock ${LIBRARIES} pthread)

add_test(TestCpu test_cpu)

if(TEST_CUDA)
  cuda_add_executable(test_gpu test/test_multiply.cu.cc)
  target_link_libraries(test_gpu ${GTEST_LIBRARIES} gmock ${LIBRARIES} pthread)
  add_test(TestGpu test_gpu)
endif(TEST_CUDA)