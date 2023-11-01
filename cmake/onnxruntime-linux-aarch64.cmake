# Copyright (c)  2022-2023  Xiaomi Corporation
message(STATUS "CMAKE_SYSTEM_NAME: ${CMAKE_SYSTEM_NAME}")
message(STATUS "CMAKE_SYSTEM_PROCESSOR: ${CMAKE_SYSTEM_PROCESSOR}")

if(NOT CMAKE_SYSTEM_NAME STREQUAL Linux)
  message(FATAL_ERROR "This file is for Linux only. Given: ${CMAKE_SYSTEM_NAME}")
endif()

if(NOT CMAKE_SYSTEM_PROCESSOR STREQUAL aarch64)
  message(FATAL_ERROR "This file is for aarch64 only. Given: ${CMAKE_SYSTEM_PROCESSOR}")
endif()

if(NOT BUILD_SHARED_LIBS)
  message(FATAL_ERROR "This file is for building shared libraries. BUILD_SHARED_LIBS: ${BUILD_SHARED_LIBS}")
endif()

set(onnxruntime_URL "https://github.com/csukuangfj/onnxruntime-libs/releases/download/v1.16.1/onnxruntime-linux-aarch64-1.16.1.tgz")
set(onnxruntime_URL2 "https://huggingface.co/csukuangfj/onnxruntime-libs/resolve/main/onnxruntime-linux-aarch64-1.16.1.tgz")
set(onnxruntime_HASH "SHA256=f10851b62eb44f9e811134737e7c6edd15733d2c1549cb6ce403808e9c047385")

# If you don't have access to the Internet,
# please download onnxruntime to one of the following locations.
# You can add more if you want.
set(possible_file_locations
  $ENV{HOME}/Downloads/onnxruntime-linux-aarch64-1.16.1.tgz
  ${PROJECT_SOURCE_DIR}/onnxruntime-linux-aarch64-1.16.1.tgz
  ${PROJECT_BINARY_DIR}/onnxruntime-linux-aarch64-1.16.1.tgz
  /tmp/onnxruntime-linux-aarch64-1.16.1.tgz
  /star-fj/fangjun/download/github/onnxruntime-linux-aarch64-1.16.1.tgz
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

find_library(location_onnxruntime onnxruntime
  PATHS
  "${onnxruntime_SOURCE_DIR}/lib"
  NO_CMAKE_SYSTEM_PATH
)

message(STATUS "location_onnxruntime: ${location_onnxruntime}")

add_library(onnxruntime SHARED IMPORTED)

set_target_properties(onnxruntime PROPERTIES
  IMPORTED_LOCATION ${location_onnxruntime}
  INTERFACE_INCLUDE_DIRECTORIES "${onnxruntime_SOURCE_DIR}/include"
)

file(GLOB onnxruntime_lib_files "${onnxruntime_SOURCE_DIR}/lib/libonnxruntime*")
message(STATUS "onnxruntime lib files: ${onnxruntime_lib_files}")
install(FILES ${onnxruntime_lib_files} DESTINATION lib)
