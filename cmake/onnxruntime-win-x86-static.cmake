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
set(ONNXRUNTIME_HASH_MT_Release "SHA256=809037d37d2a94b67e426f4bf049a64ce5674f16211b0a92358c0047ac2a4a45")
set(ONNXRUNTIME_HASH_MT_Debug "SHA256=5e5e6a6a3d1dbcea2c126dea4ebf14521effc3c04c12c2bdb44a1939830b57a5")
set(ONNXRUNTIME_HASH_MT_RelWithDebInfo "SHA256=b399f4f3cdff0193623efb738ed2170384f63d6290c083dac6941490fcc7aa1b")
set(ONNXRUNTIME_HASH_MT_MinSizeRel "SHA256=eb326bfd9f44a6507b6dce525250c60be82f0f7d366c5a36d6b77f1701d247dd")

# Hashes for dynamic CRT (/MD)
set(ONNXRUNTIME_HASH_MD_Release "SHA256=ba5ac27b62ae160727affe10cf9ad6bf79e08a0cf1d12032262c8b33291ec007")
set(ONNXRUNTIME_HASH_MD_Debug "SHA256=6c13a20f32353441d841b6024291c022fe0a94d04b10c791b3e253424185b5cf")
set(ONNXRUNTIME_HASH_MD_RelWithDebInfo "SHA256=fcc769b7a5e898fd6abdee8d606ce214e0d0d22c777c81425cb971c8e8a129d5")
set(ONNXRUNTIME_HASH_MD_MinSizeRel "SHA256=6d1fe00a7b411180dc11555bd7c9cc5dd3caf6d90eaffdedb71b0d0c21936a68")

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
set(onnxruntime_filename "onnxruntime-win-x86-static_lib-${onnxruntime_crt}-${CMAKE_BUILD_TYPE}-1.27.0.tar.bz2")
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
