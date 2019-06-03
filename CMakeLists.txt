CMAKE_MINIMUM_REQUIRED( VERSION 2.8 )
project(example)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -Ofast -Wall -Wextra ")
enable_testing()

list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)

find_package(GTest REQUIRED)
include_directories(${GTEST_INCLUDE_DIRS})

find_package(CUDA QUIET)
include(CudaHelper)

include_directories(${CMAKE_CURRENT_SOURCE_DIR})

add_library(multiply_cpu SHARED src/multiply/multiply_cpu.cc )
list(APPEND LIBRARIES "multiply_cpu")

if(CUDA_FOUND)
  cuda_add_library(multiply_gpu SHARED src/multiply/multiply_gpu.cu )
  list(APPEND LIBRARIES "multiply_gpu")

  cuda_add_executable(sharedmemory src/sharedmemory.cu )
  cuda_add_executable(tune src/tune.cu )
endif(CUDA_FOUND)

add_executable(multiply src/multiply.cc )
target_link_libraries(multiply LINK_PUBLIC ${LIBRARIES})

# TESTS
add_executable(test_cpu test/test_multiply.cc)
target_link_libraries(test_cpu ${GTEST_LIBRARIES} multiply_cpu pthread)

add_test(TestCpu test_cpu)

if(TEST_CUDA)
  cuda_add_executable(test_gpu test/test_multiply.cu)
  target_link_libraries(test_gpu ${GTEST_LIBRARIES} multiply_gpu pthread)
  add_test(TestGpu test_gpu)
endif(TEST_CUDA)