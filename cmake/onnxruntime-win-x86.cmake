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
set(ONNXRUNTIME_HASH_MT_Release "SHA256=8837ccd31e0e20ea2e53b99e26ae833d7162cf67dcf3f581e023d7c9758bad0e")
set(ONNXRUNTIME_HASH_MT_Debug "SHA256=ccb63018439961f41feac0432272456490789d14255e9540d86da573d7908d45")
set(ONNXRUNTIME_HASH_MT_RelWithDebInfo "SHA256=235ebd7f29ac74cb8b792a05b8f2159d68db154b187882d0e7b1d7e8bdbb2fb5")
set(ONNXRUNTIME_HASH_MT_MinSizeRel "SHA256=87a9f1a7f8215bd782f7284eacda71029c4a38f82e379086cef1614403246b93")

# Hashes for dynamic CRT (/MD)
set(ONNXRUNTIME_HASH_MD_Release "SHA256=a87150a811a2257d4ca4c173ba9f5783b28e8cecbcf463c402834ba538ec795d")
set(ONNXRUNTIME_HASH_MD_Debug "SHA256=f68d3d9c82c620d61afcd6eecfa28dff551523ad5a8d46f85ae34d7c8723ce90")
set(ONNXRUNTIME_HASH_MD_RelWithDebInfo "SHA256=88346c7727088e3c703abad1c7cab33bdf333cf48bc5ec4adef428c93a320e47")
set(ONNXRUNTIME_HASH_MD_MinSizeRel "SHA256=e085084b1be91375445c45a3ad822be40ace01e48ac8ef09388629274dcc4266")

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
set(onnxruntime_filename "onnxruntime-win-x86-${onnxruntime_crt}-${CMAKE_BUILD_TYPE}-1.27.1.tar.bz2")
set(onnxruntime_URL  "https://github.com/csukuangfj/onnxruntime-libs/releases/download/v1.27.1/${onnxruntime_filename}")

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
install(FILES "${onnxruntime_SOURCE_DIR}/lib/onnxruntime.lib" DESTINATION lib)
