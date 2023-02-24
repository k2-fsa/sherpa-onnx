cmake_minimum_required(VERSION 3.13 FATAL_ERROR)
project(sherpa-onnx)

set(SHERPA_ONNX_VERSION "1.1")

# Disable warning about
#
# "The DOWNLOAD_EXTRACT_TIMESTAMP option was not given and policy CMP0135 is
#  not set.
if (CMAKE_VERSION VERSION_GREATER_EQUAL "3.24.0")
  cmake_policy(SET CMP0135 NEW)
endif()

option(SHERPA_ONNX_ENABLE_PYTHON "Whether to build Python" OFF)
option(SHERPA_ONNX_ENABLE_TESTS "Whether to build tests" OFF)
option(SHERPA_ONNX_ENABLE_CHECK "Whether to build with assert" ON)
option(BUILD_SHARED_LIBS "Whether to build shared libraries" OFF)
option(SHERPA_ONNX_ENABLE_PORTAUDIO "Whether to build with portaudio" ON)
option(SHERPA_ONNX_ENABLE_JNI "Whether to build JNI internface" OFF)
option(SHERPA_ONNX_ENABLE_C_API "Whether to build C API" ON)
option(SHERPA_ONNX_ENABLE_WEBSOCKET "Whether to build webscoket server/client" ON)

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")

set(CMAKE_SKIP_BUILD_RPATH FALSE)
set(BUILD_RPATH_USE_ORIGIN TRUE)
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)

if(NOT APPLE)
  set(SHERPA_ONNX_RPATH_ORIGIN "$ORIGIN")
else()
  set(SHERPA_ONNX_RPATH_ORIGIN "@loader_path")
endif()

set(CMAKE_INSTALL_RPATH ${SHERPA_ONNX_RPATH_ORIGIN})
set(CMAKE_BUILD_RPATH ${SHERPA_ONNX_RPATH_ORIGIN})

if(WIN32 AND BUILD_SHARED_LIBS)
  message(STATUS "Set BUILD_SHARED_LIBS to OFF for windows")
  set(BUILD_SHARED_LIBS OFF)
endif()

if(NOT CMAKE_BUILD_TYPE)
  message(STATUS "No CMAKE_BUILD_TYPE given, default to Release")
  set(CMAKE_BUILD_TYPE Release)
endif()

if(DEFINED ANDROID_ABI)
  message(STATUS "Set SHERPA_ONNX_ENABLE_JNI to ON for Android")
  set(SHERPA_ONNX_ENABLE_JNI ON CACHE BOOL "" FORCE)
endif()

message(STATUS "CMAKE_BUILD_TYPE: ${CMAKE_BUILD_TYPE}")
message(STATUS "CMAKE_INSTALL_PREFIX: ${CMAKE_INSTALL_PREFIX}")
message(STATUS "BUILD_SHARED_LIBS ${BUILD_SHARED_LIBS}")
message(STATUS "SHERPA_ONNX_ENABLE_PYTHON ${SHERPA_ONNX_ENABLE_PYTHON}")
message(STATUS "SHERPA_ONNX_ENABLE_TESTS ${SHERPA_ONNX_ENABLE_TESTS}")
message(STATUS "SHERPA_ONNX_ENABLE_CHECK ${SHERPA_ONNX_ENABLE_CHECK}")
message(STATUS "SHERPA_ONNX_ENABLE_PORTAUDIO ${SHERPA_ONNX_ENABLE_PORTAUDIO}")
message(STATUS "SHERPA_ONNX_ENABLE_JNI ${SHERPA_ONNX_ENABLE_JNI}")
message(STATUS "SHERPA_ONNX_ENABLE_C_API ${SHERPA_ONNX_ENABLE_C_API}")
message(STATUS "SHERPA_ONNX_ENABLE_WEBSOCKET ${SHERPA_ONNX_ENABLE_WEBSOCKET}")

set(CMAKE_CXX_STANDARD 14 CACHE STRING "The C++ version to be used.")
set(CMAKE_CXX_EXTENSIONS OFF)

include(CheckIncludeFileCXX)
check_include_file_cxx(alsa/asoundlib.h SHERPA_ONNX_HAS_ALSA)
if(SHERPA_ONNX_HAS_ALSA)
  add_definitions(-DSHERPA_ONNX_ENABLE_ALSA=1)
endif()

check_include_file_cxx(cxxabi.h SHERPA_ONNX_HAVE_CXXABI_H)
check_include_file_cxx(execinfo.h SHERPA_ONNX_HAVE_EXECINFO_H)

list(APPEND CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/cmake/Modules)
list(APPEND CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/cmake)

include(kaldi-native-fbank)
include(onnxruntime)

if(SHERPA_ONNX_ENABLE_PORTAUDIO)
  include(portaudio)
endif()

if(SHERPA_ONNX_ENABLE_PYTHON)
  include(pybind11)
endif()

if(SHERPA_ONNX_ENABLE_TESTS)
  enable_testing()
  include(googletest)
endif()

if(SHERPA_ONNX_ENABLE_WEBSOCKET)
  include(websocketpp)
  include(asio)
endif()

add_subdirectory(sherpa-onnx)

if(SHERPA_ONNX_ENABLE_C_API)
  add_subdirectory(c-api-examples)
endif()
