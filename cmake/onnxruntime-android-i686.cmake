# Copyright (c)  2026  Xiaomi Corporation
message(STATUS "CMAKE_SYSTEM_NAME: ${CMAKE_SYSTEM_NAME}")
message(STATUS "CMAKE_SYSTEM_PROCESSOR: ${CMAKE_SYSTEM_PROCESSOR}")

if(NOT CMAKE_SYSTEM_NAME STREQUAL Android)
  message(FATAL_ERROR "This file is for Android only. Given: ${CMAKE_SYSTEM_NAME}")
endif()

if(NOT CMAKE_SYSTEM_PROCESSOR STREQUAL i686)
  message(FATAL_ERROR "This file is for i686 only. Given: ${CMAKE_SYSTEM_PROCESSOR}")
endif()

if(NOT BUILD_SHARED_LIBS)
  message(FATAL_ERROR "This file is for building shared libraries. BUILD_SHARED_LIBS: ${BUILD_SHARED_LIBS}")
endif()

set(onnxruntime_URL  "https://github.com/csukuangfj/onnxruntime-libs/releases/download/v1.28.2/onnxruntime-android-1.28.2.zip")
set(onnxruntime_HASH "SHA256=01518867f78241138b6aa25925802e843a4fa9085af8d303d49e35bbb52aff4d")

set(possible_file_locations
  $ENV{HOME}/Downloads/onnxruntime-android-1.28.2.zip
  ${CMAKE_SOURCE_DIR}/onnxruntime-android-1.28.2.zip
  ${CMAKE_BINARY_DIR}/onnxruntime-android-1.28.2.zip
  /tmp/onnxruntime-android-1.28.2.zip
)

foreach(f IN LISTS possible_file_locations)
  if(EXISTS ${f})
    set(onnxruntime_URL  "${f}")
    file(TO_CMAKE_PATH "${onnxruntime_URL}" onnxruntime_URL)
    message(STATUS "Found local downloaded onnxruntime: ${onnxruntime_URL}")
    break()
  endif()
endforeach()

FetchContent_Declare(onnxruntime
  URL
    ${onnxruntime_URL}
  URL_HASH          ${onnxruntime_HASH}
)

FetchContent_GetProperties(onnxruntime)
if(NOT onnxruntime_POPULATED)
  message(STATUS "Downloading onnxruntime from ${onnxruntime_URL}")
  FetchContent_Populate(onnxruntime)
endif()
message(STATUS "onnxruntime is downloaded to ${onnxruntime_SOURCE_DIR}")

set(location_onnxruntime "${onnxruntime_SOURCE_DIR}/jni/x86/libonnxruntime.so")

message(STATUS "location_onnxruntime: ${location_onnxruntime}")

add_library(onnxruntime SHARED IMPORTED)

set_target_properties(onnxruntime PROPERTIES
  IMPORTED_LOCATION ${location_onnxruntime}
  INTERFACE_INCLUDE_DIRECTORIES "${onnxruntime_SOURCE_DIR}/headers"
)

file(GLOB onnxruntime_lib_files "${onnxruntime_SOURCE_DIR}/jni/x86/libonnxruntime*")
message(STATUS "onnxruntime lib files: ${onnxruntime_lib_files}")
install(FILES ${onnxruntime_lib_files} DESTINATION lib)
