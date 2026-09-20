# The Flutter tooling requires that developers have a version of Visual Studio
# installed that includes CMake 3.14 or later. You should not increase this
# version, as doing so will cause the plugin to fail to compile for some
# customers of the plugin.
cmake_minimum_required(VERSION 3.14)

# Project-level configuration.
set(PROJECT_NAME "sherpa_onnx_windows")
project(${PROJECT_NAME} LANGUAGES CXX)

# The package ships pre-built libraries for two architectures:
#
#   windows/x64/*.dll    x64
#   windows/arm64/*.dll  arm64
#
# Pick the set matching the architecture being built FOR. Flutter names it
# in FLUTTER_TARGET_PLATFORM (windows-x64 or windows-arm64) and also passes
# it to Visual Studio as the generator platform (-A ARM64), which is what a
# plain CMake configure sees. CMAKE_SYSTEM_PROCESSOR is deliberately not
# used: it describes the machine running CMake, so an x64 build made on an
# arm64 machine would ship the wrong libraries.
if(FLUTTER_TARGET_PLATFORM STREQUAL "windows-arm64"
   OR CMAKE_GENERATOR_PLATFORM MATCHES "^ARM64$")
  set(sherpa_onnx_windows_lib_dir "${CMAKE_CURRENT_SOURCE_DIR}/arm64")
else()
  set(sherpa_onnx_windows_lib_dir "${CMAKE_CURRENT_SOURCE_DIR}/x64")
endif()

# List of absolute paths to libraries that should be bundled with the plugin.
# This list could contain prebuilt libraries, or libraries created by an
# external build triggered from this build file.
set(sherpa_onnx_windows_bundled_libraries
  "${sherpa_onnx_windows_lib_dir}/sherpa-onnx-c-api.dll"
  "${sherpa_onnx_windows_lib_dir}/onnxruntime.dll"
  PARENT_SCOPE
)
