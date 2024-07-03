# The Flutter tooling requires that developers have CMake 3.10 or later
# installed. You should not increase this version, as doing so will cause
# the plugin to fail to compile for some customers of the plugin.
cmake_minimum_required(VERSION 3.10)

# Project-level configuration.
set(PROJECT_NAME "sherpa_onnx_linux")
project(${PROJECT_NAME} LANGUAGES CXX)

# List of absolute paths to libraries that should be bundled with the plugin.
# This list could contain prebuilt libraries, or libraries created by an
# external build triggered from this build file.
set(sherpa_onnx_linux_bundled_libraries
  "${CMAKE_CURRENT_SOURCE_DIR}/libsherpa-onnx-c-api.so"
  "${CMAKE_CURRENT_SOURCE_DIR}/libsherpa-onnx-core.so"
  "${CMAKE_CURRENT_SOURCE_DIR}/libkaldi-decoder-core.so"
  "${CMAKE_CURRENT_SOURCE_DIR}/libsherpa-onnx-kaldifst-core.so"
  "${CMAKE_CURRENT_SOURCE_DIR}/libsherpa-onnx-fstfar.so"
  "${CMAKE_CURRENT_SOURCE_DIR}/libsherpa-onnx-fst.so"
  "${CMAKE_CURRENT_SOURCE_DIR}/libkaldi-native-fbank-core.so"
  "${CMAKE_CURRENT_SOURCE_DIR}/libpiper_phonemize.so"
  "${CMAKE_CURRENT_SOURCE_DIR}/libespeak-ng.so"
  "${CMAKE_CURRENT_SOURCE_DIR}/libucd.so"
  "${CMAKE_CURRENT_SOURCE_DIR}/libonnxruntime.so"
  "${CMAKE_CURRENT_SOURCE_DIR}/libssentencepiece_core.so"
  PARENT_SCOPE
)
