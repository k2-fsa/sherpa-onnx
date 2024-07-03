# The Flutter tooling requires that developers have CMake 3.10 or later
# installed. You should not increase this version, as doing so will cause
# the plugin to fail to compile for some customers of the plugin.
cmake_minimum_required(VERSION 3.10)

project(sherpa_onnx_android_library VERSION 0.0.1 LANGUAGES C)

add_library(sherpa_onnx_android SHARED
  "sherpa_onnx_android.c"
)

set_target_properties(sherpa_onnx_android PROPERTIES
  PUBLIC_HEADER sherpa_onnx_android.h
  OUTPUT_NAME "sherpa_onnx_android"
)

target_compile_definitions(sherpa_onnx_android PUBLIC DART_SHARED_LIB)
