# Copyright (c)  2022-2023  Xiaomi Corporation
message(STATUS "CMAKE_SYSTEM_NAME: ${CMAKE_SYSTEM_NAME}")
message(STATUS "CMAKE_SYSTEM_PROCESSOR: ${CMAKE_SYSTEM_PROCESSOR}")

if(NOT CMAKE_SYSTEM_NAME STREQUAL Linux)
  message(FATAL_ERROR "This file is for Linux only. Given: ${CMAKE_SYSTEM_NAME}")
endif()

if(NOT CMAKE_SYSTEM_PROCESSOR STREQUAL x86_64)
  message(FATAL_ERROR "This file is for x86_64 only. Given: ${CMAKE_SYSTEM_PROCESSOR}")
endif()

if(BUILD_SHARED_LIBS)
  message(FATAL_ERROR "This file is for building static libraries. BUILD_SHARED_LIBS: ${BUILD_SHARED_LIBS}")
endif()

set(onnxruntime_URL  "https://github.com/csukuangfj/onnxruntime-libs/releases/download/v1.17.1/onnxruntime-linux-x64-static_lib-1.17.1-glibc2_17.zip")
set(onnxruntime_URL2 "https://hf-mirror.com/csukuangfj/onnxruntime-libs/resolve/main/onnxruntime-linux-x64-static_lib-1.17.1-glibc2_17.zip")
set(onnxruntime_HASH "SHA256=b646beeb983de843a267096d4457d832f93089f5e7264fd54b48cff207cb2068")

# If you don't have access to the Internet,
# please download onnxruntime to one of the following locations.
# You can add more if you want.
set(possible_file_locations
  $ENV{HOME}/Downloads/onnxruntime-linux-x64-static_lib-1.17.1-glibc2_17.zip
  ${CMAKE_SOURCE_DIR}/onnxruntime-linux-x64-static_lib-1.17.1-glibc2_17.zip
  ${CMAKE_BINARY_DIR}/onnxruntime-linux-x64-static_lib-1.17.1-glibc2_17.zip
  /tmp/onnxruntime-linux-x64-static_lib-1.17.1-glibc2_17.zip
  /star-fj/fangjun/download/github/onnxruntime-linux-x64-static_lib-1.17.1-glibc2_17.zip
)

foreach(f IN LISTS possible_file_locations)
  if(EXISTS ${f})
    set(onnxruntime_URL  "${f}")
    file(TO_CMAKE_PATH "${onnxruntime_URL}" onnxruntime_URL)
    message(STATUS "Found local downloaded onnxruntime: ${onnxruntime_URL}")
    set(onnxruntime_URL2)
    break()
  endif()
endforeach()

FetchContent_Declare(onnxruntime
  URL
    ${onnxruntime_URL}
    ${onnxruntime_URL2}
  URL_HASH          ${onnxruntime_HASH}
)

FetchContent_GetProperties(onnxruntime)
if(NOT onnxruntime_POPULATED)
  message(STATUS "Downloading onnxruntime from ${onnxruntime_URL}")
  FetchContent_Populate(onnxruntime)
endif()
message(STATUS "onnxruntime is downloaded to ${onnxruntime_SOURCE_DIR}")

# for static libraries, we use onnxruntime_lib_files directly below
include_directories(${onnxruntime_SOURCE_DIR}/include)

file(GLOB onnxruntime_lib_files "${onnxruntime_SOURCE_DIR}/lib/lib*.a")

set(onnxruntime_lib_files ${onnxruntime_lib_files} PARENT_SCOPE)

message(STATUS "onnxruntime lib files: ${onnxruntime_lib_files}")
install(FILES ${onnxruntime_lib_files} DESTINATION lib)
