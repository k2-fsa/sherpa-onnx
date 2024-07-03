# The Flutter tooling requires that developers have CMake 3.10 or later
# installed. You should not increase this version, as doing so will cause
# the plugin to fail to compile for some customers of the plugin.
cmake_minimum_required(VERSION 3.10)

project(sherpa_onnx_linux_library VERSION 0.0.1 LANGUAGES C)

add_library(sherpa_onnx_linux SHARED
  "sherpa_onnx_linux.c"
)

set_target_properties(sherpa_onnx_linux PROPERTIES
  PUBLIC_HEADER sherpa_onnx_linux.h
  OUTPUT_NAME "sherpa_onnx_linux"
)

target_compile_definitions(sherpa_onnx_linux PUBLIC DART_SHARED_LIB)
