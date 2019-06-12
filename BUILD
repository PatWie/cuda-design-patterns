package(default_visibility=["//visibility:public"])
load("//:cuda.bzl", "cuda_binary")

cc_library(
    name="multiply",
    srcs=[
        "src/multiply/multiply.cc",
        "src/multiply/multiply_cpu.cc"
    ],
    hdrs=[
        "include/multiply/multiply.h",
        "include/cuda_utils.h"
    ],
)

# TODO(patwie): typed tests fail
# cc_test(
#     name="multiply-test",
#     srcs=[
#         "test/test_multiply_impl.h",
#         "test/test_multiply.cc",
#     ],
#     copts=[
#         "-Iexternal/gtest/include",
#         "-Iexternal/gmock/include"
#     ],
#     deps=[
#         ":multiply",
#         "@gtest//:gtest",
#     ]
# )

cc_binary(
    name="multiply-example",
    srcs=[
        "src/multiply.cc"
    ],
    deps=[
        ":multiply",
    ],
)

cuda_binary(
    name="sharedmemory-example",
    includes=[
        "/usr/local/cuda/include",
        "."
    ],
    hdrs=[
        "include/cuda_utils.h",
        "include/cuda_index.h",
        "include/test/multiply.h",
    ],
    flags="-std=c++11",
    srcs=[
        "src/sharedmemory.cu.cc"
    ],
)

cuda_binary(
    name="tune-example",
    includes=[
        "/usr/local/cuda/include",
        "."
    ],
    hdrs=[
        "include/cuda_utils.h",
        "include/test/multiply.h",
    ],
    flags="-std=c++11",
    srcs=[
        "src/tune.cu.cc"
    ],

)
