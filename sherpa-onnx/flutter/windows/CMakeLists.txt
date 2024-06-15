# The Flutter tooling requires that developers have a version of Visual Studio
# installed that includes CMake 3.14 or later. You should not increase this
# version, as doing so will cause the plugin to fail to compile for some
# customers of the plugin.
cmake_minimum_required(VERSION 3.14)

# Project-level configuration.
set(PROJECT_NAME "sherpa_onnx")
project(${PROJECT_NAME} LANGUAGES CXX)

# Invoke the build for native code shared with the other target platforms.
# This can be changed to accommodate different builds.
# add_subdirectory("${CMAKE_CURRENT_SOURCE_DIR}/../src" "${CMAKE_CURRENT_BINARY_DIR}/shared")

# List of absolute paths to libraries that should be bundled with the plugin.
# This list could contain prebuilt libraries, or libraries created by an
# external build triggered from this build file.
set(sherpa_onnx_bundled_libraries
  "${CMAKE_CURRENT_SOURCE_DIR}/sherpa-onnx-c-api.dll"
  "${CMAKE_CURRENT_SOURCE_DIR}/sherpa-onnx-core.dll"
  "${CMAKE_CURRENT_SOURCE_DIR}/kaldi-decoder-core.dll"
  "${CMAKE_CURRENT_SOURCE_DIR}/sherpa-onnx-kaldifst-core.dll"
  "${CMAKE_CURRENT_SOURCE_DIR}/sherpa-onnx-fstfar.dll"
  "${CMAKE_CURRENT_SOURCE_DIR}/sherpa-onnx-fst.dll"
  "${CMAKE_CURRENT_SOURCE_DIR}/kaldi-native-fbank-core.dll"
  "${CMAKE_CURRENT_SOURCE_DIR}/piper_phonemize.dll"
  "${CMAKE_CURRENT_SOURCE_DIR}/espeak-ng.dll"
  "${CMAKE_CURRENT_SOURCE_DIR}/ucd.dll"
  "${CMAKE_CURRENT_SOURCE_DIR}/onnxruntime.dll"
  "${CMAKE_CURRENT_SOURCE_DIR}/ssentencepiece_core.dll"
  PARENT_SCOPE
)
