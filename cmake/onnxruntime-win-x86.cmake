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

if(NOT BUILD_SHARED_LIBS)
  message(FATAL_ERROR "This file is for building shared libraries. BUILD_SHARED_LIBS: ${BUILD_SHARED_LIBS}")
endif()

# Hashes for static CRT (/MT)
set(ONNXRUNTIME_HASH_MT_Release "SHA256=07536b6b0c3929df8a41352331357daac99deecf49a7856493c7d24dc036b071")
set(ONNXRUNTIME_HASH_MT_Debug "SHA256=6f9146c969f2db41049d604a3a6f922a61a7a18e2116d539bce9578da7092037")
set(ONNXRUNTIME_HASH_MT_RelWithDebInfo "SHA256=d9a40560ca39425fc19c599e2b33b39459660deb030c0dc5ef83ddb7bf5f58d0")
set(ONNXRUNTIME_HASH_MT_MinSizeRel "SHA256=c5dc5570a8144b152592de69ecf4d2aaa4557a62dc4c47e24f480e26c831bd65")

# Hashes for dynamic CRT (/MD)
set(ONNXRUNTIME_HASH_MD_Release "SHA256=ce1519a7934204cbf9f3431ba1d67ea1a8e838f743245ac96db9faaaf150581c")
set(ONNXRUNTIME_HASH_MD_Debug "SHA256=d35d5b5bee5a0483f16e845783902c686c9a186c71c17bcf20d6887a734a6ad9")
set(ONNXRUNTIME_HASH_MD_RelWithDebInfo "SHA256=e4df40832040419d9b5e7d983420c51e3095987bd20314c9d7a90b4759df2991")
set(ONNXRUNTIME_HASH_MD_MinSizeRel "SHA256=4ea6d745466f3623a13c0159f09dcf50d6b34e302b219cfc396c6af7122b7b39")

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
set(onnxruntime_filename "onnxruntime-win-x86-${onnxruntime_crt}-${CMAKE_BUILD_TYPE}-1.23.2.tar.bz2")
set(onnxruntime_URL  "https://github.com/csukuangfj/onnxruntime-libs/releases/download/v1.23.2/${onnxruntime_filename}")

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

set_property(TARGET onnxruntime
  PROPERTY
    IMPORTED_IMPLIB "${onnxruntime_SOURCE_DIR}/lib/onnxruntime.lib"
)

file(COPY ${onnxruntime_SOURCE_DIR}/lib/onnxruntime.dll
  DESTINATION
    ${CMAKE_BINARY_DIR}/bin/${CMAKE_BUILD_TYPE}
)

file(GLOB onnxruntime_lib_files "${onnxruntime_SOURCE_DIR}/lib/*.dll")

message(STATUS "onnxruntime lib files: ${onnxruntime_lib_files}")

install(FILES ${onnxruntime_lib_files} DESTINATION lib)
install(FILES ${onnxruntime_lib_files} DESTINATION bin)
