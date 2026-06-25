# Copyright (c)  2022-2023  Xiaomi Corporation
message(STATUS "CMAKE_SYSTEM_NAME: ${CMAKE_SYSTEM_NAME}")
message(STATUS "CMAKE_SYSTEM_PROCESSOR: ${CMAKE_SYSTEM_PROCESSOR}")
message(STATUS "CMAKE_VS_PLATFORM_NAME: ${CMAKE_VS_PLATFORM_NAME}")

if(NOT CMAKE_SYSTEM_NAME STREQUAL Windows)
  message(FATAL_ERROR "This file is for Windows only. Given: ${CMAKE_SYSTEM_NAME}")
endif()

if(NOT (CMAKE_VS_PLATFORM_NAME STREQUAL ARM64 OR CMAKE_VS_PLATFORM_NAME STREQUAL arm64))
  message(FATAL_ERROR "This file is for Windows arm64 only. Given: ${CMAKE_VS_PLATFORM_NAME}")
endif()

if(BUILD_SHARED_LIBS)
  message(FATAL_ERROR "This file is for building static libraries. BUILD_SHARED_LIBS: ${BUILD_SHARED_LIBS}")
endif()

# Hashes for static CRT (/MT)
set(ONNXRUNTIME_HASH_MT_Release "SHA256=f06d31316230095656a68f42db20aba81ece0edb1d437881efad9479ee5f5d82")
set(ONNXRUNTIME_HASH_MT_Debug "SHA256=8ef5f2af576505e0edeba3fc7bddb7298e4db09bdb1a38d9699d5f3d17fe0093")
set(ONNXRUNTIME_HASH_MT_RelWithDebInfo "SHA256=40a67c34a22a9761ea076402a78acfe019503a623a77b4dc918f61121a7d07dd")
set(ONNXRUNTIME_HASH_MT_MinSizeRel "SHA256=8142133abcc34d07bbff22004dc457b2feba55ff5bbad452424cb2942765fb02")

# Hashes for dynamic CRT (/MD)
set(ONNXRUNTIME_HASH_MD_Release "SHA256=985c088ff0771c8397ed8464ceed2f8037fca537345d3b0a1dd67ebda58a3e8e")
set(ONNXRUNTIME_HASH_MD_Debug "SHA256=e7c95cc511a4d459d82059ff59c58890e363046dd4b0e043114e5a77d60c8a9b")
set(ONNXRUNTIME_HASH_MD_RelWithDebInfo "SHA256=1f4752659a0baf906ef3094ae4f0f1b0967697af43699ce5d965cc61cd6889d8")
set(ONNXRUNTIME_HASH_MD_MinSizeRel "SHA256=459cf4a92d16080a4a87d5a993b25adc3fc34b86bfa909779139c0be4901c78b")

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
set(onnxruntime_filename "onnxruntime-win-arm64-static_lib-${onnxruntime_crt}-${CMAKE_BUILD_TYPE}-1.27.0.tar.bz2")
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
