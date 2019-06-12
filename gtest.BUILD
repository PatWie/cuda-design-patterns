cc_library(
    visibility=["//visibility:public"],
    name="gtest",
    srcs=glob(
        include=[
            "googletest/src/*.cc",
            "googletest/src/*.h",
            "googletest/include/gtest/**/*.h",
            "googlemock/src/*.cc",
            "googlemock/include/gmock/**/*.h",
        ],
        exclude=[
            "googletest/src/gtest-all.cc",
            "googletest/src/gtest_main.cc",
            "googlemock/src/gmock-all.cc",
            "googlemock/src/gmock_main.cc",
        ],
    ),
    hdrs=glob([
        "googletest/include/gtest/*.h",
        "googlemock/include/gmock/*.h",
    ]),
    copts=select({
        "//conditions:default": ["-pthread -DGTEST_HAS_TYPED_TEST_P"],
    }),
    defines=select({
        "//conditions:default": [],
    }),
    includes=[
        "googlemock",
        "googlemock/include",
        "googletest",
        "googletest/include",
    ],
    linkopts=select({
        "//conditions:default": ["-pthread"],
    }),
    deps=select({
        "//conditions:default": [],
    }),
    features=select({
        "//conditions:default": [],
    })
)
