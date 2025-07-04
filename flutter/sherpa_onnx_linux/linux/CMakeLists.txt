# The Flutter tooling requires that developers have CMake 3.10 or later
# installed. You should not increase this version, as doing so will cause
# the plugin to fail to compile for some customers of the plugin.
cmake_minimum_required(VERSION 3.10)

# Project-level configuration.
set(PROJECT_NAME "sherpa_onnx_linux")
project(${PROJECT_NAME} LANGUAGES CXX)

if(CMAKE_SYSTEM_PROCESSOR STREQUAL "x86_64")
  set(LIB_ARCH_DIR "x64")
elseif(CMAKE_SYSTEM_PROCESSOR STREQUAL "aarch64")
  set(LIB_ARCH_DIR "aarch64")
else()
  message(FATAL_ERROR "Unsupported arch: ${CMAKE_SYSTEM_PROCESSOR}")
endif()

# List of absolute paths to libraries that should be bundled with the plugin.
# This list could contain prebuilt libraries, or libraries created by an
# external build triggered from this build file.
set(sherpa_onnx_linux_bundled_libraries
  "${CMAKE_CURRENT_SOURCE_DIR}/${LIB_ARCH_DIR}/libsherpa-onnx-c-api.so"
  "${CMAKE_CURRENT_SOURCE_DIR}/${LIB_ARCH_DIR}/libonnxruntime.so"
  PARENT_SCOPE
)
