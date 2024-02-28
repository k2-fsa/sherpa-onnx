# Copyright (c)  2022-2023  Xiaomi Corporation
message(STATUS "CMAKE_SYSTEM_NAME: ${CMAKE_SYSTEM_NAME}")
message(STATUS "CMAKE_SYSTEM_PROCESSOR: ${CMAKE_SYSTEM_PROCESSOR}")
message(STATUS "CMAKE_VS_PLATFORM_NAME: ${CMAKE_VS_PLATFORM_NAME}")

if(NOT CMAKE_SYSTEM_NAME STREQUAL Windows)
  message(FATAL_ERROR "This file is for Windows only. Given: ${CMAKE_SYSTEM_NAME}")
endif()

if(NOT (CMAKE_VS_PLATFORM_NAME STREQUAL Win32 OR CMAKE_VS_PLATFORM_NAME STREQUAL win32))
  message(FATAL_ERROR "This file is for Windows x86 only. Given: ${CMAKE_VS_PLATFORM_NAME}")
endif()

if(BUILD_SHARED_LIBS)
  message(FATAL_ERROR "This file is for building static libraries. BUILD_SHARED_LIBS: ${BUILD_SHARED_LIBS}")
endif()


set(onnxruntime_URL  "https://github.com/csukuangfj/onnxruntime-libs/releases/download/v1.17.0/onnxruntime-win-x86-static_lib-${CMAKE_BUILD_TYPE}-1.17.0.tar.bz2")
set(onnxruntime_URL2  "https://hub.nuaa.cf/csukuangfj/onnxruntime-libs/releases/download/v1.17.0/onnxruntime-win-x86-static_lib-${CMAKE_BUILD_TYPE}-1.17.0.tar.bz2")
if(CMAKE_BUILD_TYPE STREQUAL Debug)
  set(onnxruntime_HASH "SHA256=7ebf676b4cae50ae4c203ca54ae2341d7c7aeae66a965a85610b254d24effee3")
elseif(CMAKE_BUILD_TYPE STREQUAL RelWithDebInfo)
  set(onnxruntime_HASH "SHA256=bcb1dd68643baf011dc519f938c61b90556f354c6e6ce801e3f08a350d51b689")
elseif(CMAKE_BUILD_TYPE STREQUAL MinSizeRel)
  set(onnxruntime_HASH "SHA256=a73a42cbff851b1ee2ec037dd3f4254978516252bc353a7ce86d12030ba73dec")
else()
  message(FATAL_ERROR "This file is for building a debug version on Windows x86. Given ${CMAKE_BUILD_TYPE}")
endif()

# If you don't have access to the Internet,
# please download onnxruntime to one of the following locations.
# You can add more if you want.
set(possible_file_locations
  $ENV{HOME}/Downloads/onnxruntime-win-x86-static_lib-${CMAKE_BUILD_TYPE}-1.17.0.tar.bz2
  ${CMAKE_SOURCE_DIR}/onnxruntime-win-x86-static_lib-${CMAKE_BUILD_TYPE}-1.17.0.tar.bz2
  ${CMAKE_BINARY_DIR}/onnxruntime-win-x86-static_lib-${CMAKE_BUILD_TYPE}-1.17.0.tar.bz2
  /tmp/onnxruntime-win-x86-static_lib-${CMAKE_BUILD_TYPE}-1.17.0.tar.bz2
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

file(GLOB onnxruntime_lib_files "${onnxruntime_SOURCE_DIR}/lib/*.lib")
set(onnxruntime_lib_files ${onnxruntime_lib_files} PARENT_SCOPE)

message(STATUS "onnxruntime lib files: ${onnxruntime_lib_files}")
if(SHERPA_ONNX_ENABLE_PYTHON)
  install(FILES ${onnxruntime_lib_files} DESTINATION ..)
else()
  install(FILES ${onnxruntime_lib_files} DESTINATION lib)
endif()
