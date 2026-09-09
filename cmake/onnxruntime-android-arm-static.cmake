# Copyright (c)  2026  Xiaomi Corporation
message(STATUS "CMAKE_SYSTEM_NAME: ${CMAKE_SYSTEM_NAME}")
message(STATUS "CMAKE_SYSTEM_PROCESSOR: ${CMAKE_SYSTEM_PROCESSOR}")

if(NOT CMAKE_SYSTEM_NAME STREQUAL Android)
  message(FATAL_ERROR "This file is for Android only. Given: ${CMAKE_SYSTEM_NAME}")
endif()

if(NOT CMAKE_SYSTEM_PROCESSOR STREQUAL armv7-a AND NOT CMAKE_SYSTEM_PROCESSOR STREQUAL armv8l)
  message(FATAL_ERROR "This file is for armv7-a/armv8l only. Given: ${CMAKE_SYSTEM_PROCESSOR}")
endif()

if(BUILD_SHARED_LIBS)
  message(FATAL_ERROR "This file is for building static libraries. BUILD_SHARED_LIBS: ${BUILD_SHARED_LIBS}")
endif()

set(onnxruntime_URL  "https://github.com/csukuangfj/onnxruntime-libs/releases/download/v1.28.2/onnxruntime-android-armeabi-v7a-static_lib-1.28.2.zip")
set(onnxruntime_HASH "SHA256=21f8aaab5d536f397a11deeedc09c2ceecc740b45332d8a946c358e7717b5f18")

set(possible_file_locations
  $ENV{HOME}/Downloads/onnxruntime-android-armeabi-v7a-static_lib-1.28.2.zip
  ${CMAKE_SOURCE_DIR}/onnxruntime-android-armeabi-v7a-static_lib-1.28.2.zip
  ${CMAKE_BINARY_DIR}/onnxruntime-android-armeabi-v7a-static_lib-1.28.2.zip
  /tmp/onnxruntime-android-armeabi-v7a-static_lib-1.28.2.zip
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

include_directories(${onnxruntime_SOURCE_DIR}/include)

file(GLOB onnxruntime_lib_files "${onnxruntime_SOURCE_DIR}/lib/lib*.a")

set(onnxruntime_lib_files ${onnxruntime_lib_files} PARENT_SCOPE)

message(STATUS "onnxruntime lib files: ${onnxruntime_lib_files}")
install(FILES ${onnxruntime_lib_files} DESTINATION lib)
