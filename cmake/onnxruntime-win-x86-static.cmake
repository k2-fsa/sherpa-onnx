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

# Hashes for static CRT (/MT)
set(ONNXRUNTIME_HASH_MT_Release "SHA256=1d82e7bac5df9d1d1ece159344aba1f2f77e88c8391f05ebec9e0df15595cc20")
set(ONNXRUNTIME_HASH_MT_Debug "SHA256=dc3548cfa1acb6798a612cab130afc96430869b9a0fb5e712734a46e5d9b845b")
set(ONNXRUNTIME_HASH_MT_RelWithDebInfo "SHA256=89a99af7a97b485643e73428d4ad9050b16324a972de9fa567331e432e382a8e")
set(ONNXRUNTIME_HASH_MT_MinSizeRel "SHA256=2047a531fa1a23ae7ff627036cd1503446204201cc3f8f5738485b19528148b8")

# Hashes for dynamic CRT (/MD)
set(ONNXRUNTIME_HASH_MD_Release "SHA256=65265d7cc23c01a73d1640a8c00943016bae20efd34b943780e246aff8f28caa")
set(ONNXRUNTIME_HASH_MD_Debug "SHA256=af8df8749a1f8974b47f7bfb4eb8a56d3cf78f051ab37949cca73bda571365e9")
set(ONNXRUNTIME_HASH_MD_RelWithDebInfo "SHA256=45a1a50597adf2ec1152480a68e48ede9d0703a14e00a1b601f341d61a57f13a")
set(ONNXRUNTIME_HASH_MD_MinSizeRel "SHA256=44505216c815642a42c663495989307dee72d6faa0c0c2e8a6e67b670fcf79e1")

if(NOT CMAKE_BUILD_TYPE MATCHES "^(Release|Debug|RelWithDebInfo|MinSizeRel)$")
  message(FATAL_ERROR "Supported CMAKE_BUILD_TYPE values are: Release, Debug, RelWithDebInfo, MinSizeRel. Given ${CMAKE_BUILD_TYPE}")
endif()

if(SHERPA_ONNX_USE_STATIC_CRT)
  set(onnxruntime_crt "MT")
else()
  set(onnxruntime_crt "MD")
endif()

message(STATUS "Use MSVC CRT: ${onnxruntime_crt}")

set(onnxruntime_HASH "${ONNXRUNTIME_HASH_${onnxruntime_crt}_${CMAKE_BUILD_TYPE}}")
set(onnxruntime_filename "onnxruntime-win-x86-static_lib-${onnxruntime_crt}-${CMAKE_BUILD_TYPE}-1.28.2.tar.bz2")
set(onnxruntime_URL  "https://github.com/csukuangfj/onnxruntime-libs/releases/download/v1.28.2/${onnxruntime_filename}")


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
