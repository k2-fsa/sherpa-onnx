cmake_minimum_required(VERSION 3.13 FATAL_ERROR)

project(use-of-sherpa-onnx-as-a-sub-project)

if(NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/sherpa-onnx/setup.py")
  message(FATAL_ERROR "Please download the source code of sherpa-onnx and put it inside this directory")
endif()

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")

include_directories(./sherpa-onnx)
add_subdirectory(./sherpa-onnx)

add_executable(main main.cc)
target_link_libraries(main sherpa-onnx-core)
