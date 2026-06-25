# Copyright (c)  2022-2023  Xiaomi Corporation
message(STATUS "CMAKE_SYSTEM_NAME: ${CMAKE_SYSTEM_NAME}")
message(STATUS "CMAKE_SYSTEM_PROCESSOR: ${CMAKE_SYSTEM_PROCESSOR}")
message(STATUS "CMAKE_VS_PLATFORM_NAME: ${CMAKE_VS_PLATFORM_NAME}")

if(NOT CMAKE_SYSTEM_NAME STREQUAL Windows)
  message(FATAL_ERROR "This file is for Windows only. Given: ${CMAKE_SYSTEM_NAME}")
endif()

if(NOT (CMAKE_VS_PLATFORM_NAME STREQUAL X64 OR CMAKE_VS_PLATFORM_NAME STREQUAL x64))
  message(FATAL_ERROR "This file is for Windows x64 only. Given: ${CMAKE_VS_PLATFORM_NAME}")
endif()

if(BUILD_SHARED_LIBS)
  message(FATAL_ERROR "This file is for building static libraries. BUILD_SHARED_LIBS: ${BUILD_SHARED_LIBS}")
endif()

if(NOT CMAKE_BUILD_TYPE MATCHES "^(Release|Debug|RelWithDebInfo|MinSizeRel)$")
  message(FATAL_ERROR "Supported CMAKE_BUILD_TYPE values are: Release, Debug, RelWithDebInfo, MinSizeRel. Given ${CMAKE_BUILD_TYPE}")
endif()

# Hashes for static CRT (/MT)
set(ONNXRUNTIME_HASH_MT_Release "SHA256=6db0cdd3bcb208911758b7336820d19ecff0359a7003e0ec65895be701741288")
set(ONNXRUNTIME_HASH_MT_Debug "SHA256=baa0906c5e5763b0963d258528cd743e84ed95f28a4004037e4050fe891b0177")
set(ONNXRUNTIME_HASH_MT_RelWithDebInfo "SHA256=e606defb0910b411ae6e755a6f8b18bddc745a8117cb63d899ebef9a391ffb52")
set(ONNXRUNTIME_HASH_MT_MinSizeRel "SHA256=5de509681ea9b9a28535aca802ce66ed27fe5dbbacbad4dbdb1714cf15f78d0a")

# Hashes for dynamic CRT (/MD)
set(ONNXRUNTIME_HASH_MD_Release "SHA256=62a70d8d19cb567d5c816e1c071e3315a9fdd2a194a93bd5b9c8d6f81342fe9b")
set(ONNXRUNTIME_HASH_MD_Debug "SHA256=9a846f840932df91af7b6e745b864d071132163c9f4113f1f93d1dfe3aeb787e")
set(ONNXRUNTIME_HASH_MD_RelWithDebInfo "SHA256=008d80144791145b931aa63012c28b7177ffc62ea9faeaecc0de7628e108f4d9")
set(ONNXRUNTIME_HASH_MD_MinSizeRel "SHA256=2d37662e6233fafd388170557518a18aec2c0320ec1697b217ad5428e942cb3e")

if(SHERPA_ONNX_USE_STATIC_CRT)
  set(onnxruntime_crt "MT")
else()
  set(onnxruntime_crt "MD")
endif()

message(STATUS "Use MSVC CRT: ${onnxruntime_crt}")

set(onnxruntime_filename "onnxruntime-win-x64-static_lib-${onnxruntime_crt}-${CMAKE_BUILD_TYPE}-1.27.0.tar.bz2")
set(onnxruntime_HASH "${ONNXRUNTIME_HASH_${onnxruntime_crt}_${CMAKE_BUILD_TYPE}}")
set(onnxruntime_URL  "https://github.com/csukuangfj/onnxruntime-libs/releases/download/v1.27.0/${onnxruntime_filename}")

# If you don't have access to the Internet,
# please download onnxruntime to one of the following locations.
# You can add more if you want.
set(possible_file_locations
  $ENV{HOME}/Downloads/${onnxruntime_filename}
  ${CMAKE_SOURCE_DIR}/${onnxruntime_filename}
  ${CMAKE_BINARY_DIR}/${onnxruntime_filename}
  $ENV{TMP}/${onnxruntime_filename}
  $ENV{TEMP}/${onnxruntime_filename}
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
