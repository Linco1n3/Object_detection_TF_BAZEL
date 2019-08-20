# Description:

load("//tensorflow:tensorflow.bzl", "tf_cc_binary")

package(
    default_visibility = ["//tensorflow:internal"],
    licenses = ["notice"],  # Apache 2.0
)

exports_files(["LICENSE"])

tf_cc_binary(
    name = "object_detection",
    srcs = [
        "main.cpp",
    ],
    linkopts = ["-lm"],
    deps = [
        ":utils",
        "//tensorflow/cc:cc_ops",
        "//tensorflow/core:core_cpu",
        "//tensorflow/core:framework",
        "//tensorflow/core:framework_internal",
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core:tensorflow",
        "@jsoncpp_git//:jsoncpp",
        #"@boringssl//:ssl",
    ],
)

cc_library(
    name = "utils",
    srcs = [
        "utils.cpp",

    ],
    hdrs = ["utils.h",
    ],
    deps = [
        "//tensorflow/cc:cc_ops",
        "//tensorflow/core:core_cpu",
        "//tensorflow/core:framework",
        "//tensorflow/core:framework_internal",
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core:tensorflow",
    ],
    )



