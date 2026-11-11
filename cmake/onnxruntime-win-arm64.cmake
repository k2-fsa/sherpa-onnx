# Copyright (c)  2022-2024  Xiaomi Corporation
message(STATUS "CMAKE_SYSTEM_NAME: ${CMAKE_SYSTEM_NAME}")
message(STATUS "CMAKE_SYSTEM_PROCESSOR: ${CMAKE_SYSTEM_PROCESSOR}")
message(STATUS "CMAKE_VS_PLATFORM_NAME: ${CMAKE_VS_PLATFORM_NAME}")

if(NOT CMAKE_SYSTEM_NAME STREQUAL Windows)
  message(FATAL_ERROR "This file is for Windows only. Given: ${CMAKE_SYSTEM_NAME}")
endif()

if(NOT (CMAKE_VS_PLATFORM_NAME STREQUAL ARM64 OR CMAKE_VS_PLATFORM_NAME STREQUAL arm64))
  message(FATAL_ERROR "This file is for Windows arm64 only. Given: ${CMAKE_VS_PLATFORM_NAME}")
endif()

if(NOT BUILD_SHARED_LIBS)
  message(FATAL_ERROR "This file is for building shared libraries. BUILD_SHARED_LIBS: ${BUILD_SHARED_LIBS}")
endif()

if(NOT CMAKE_BUILD_TYPE MATCHES "^(Release|Debug|RelWithDebInfo|MinSizeRel)$")
  message(FATAL_ERROR "Please set CMAKE_BUILD_TYPE to Release, Debug, RelWithDebInfo or MinSizeRel")
endif()

# Hashes for static CRT (/MT)
set(ONNXRUNTIME_HASH_MT_Debug "SHA256=08771cf4fa2229b4b6240e1dfea240953cb546ff05e2bb0ff6597028ab04104d")
set(ONNXRUNTIME_HASH_MT_RelWithDebInfo "SHA256=212e423ac1bfeb5d4b32dea3d2bb4cfbf471c15db72021db55aac1dd1d67c0ba")
set(ONNXRUNTIME_HASH_MT_MinSizeRel "SHA256=d7e3e0d7deaeffbd1eaf22666695285f6e1b9d3700f9295744e6970fba0c7bfc")
set(ONNXRUNTIME_HASH_MT_Release "SHA256=59877bc38b2c1eed1375597b9da2403b314b1103693d1dd0397c00cbcad1c1ab")

# Hashes for dynamic CRT (/MD)
set(ONNXRUNTIME_HASH_MD_Debug "SHA256=c1149f620e05e66aad4cb1f53d41bc031b06867aa26a523c38681197752db006")
set(ONNXRUNTIME_HASH_MD_RelWithDebInfo "SHA256=3cbbee948a70c116627b3df80208184762d993e52da2e8ad0e923a2bb333f184")
set(ONNXRUNTIME_HASH_MD_MinSizeRel "SHA256=3e722ee4a85098f6184e7f27a017ecf5de7f4157b36132788432651de6ce7fa4")
set(ONNXRUNTIME_HASH_MD_Release "SHA256=82b68e70358030dcbe0901d3435db3398a6e34ef46e45f071088f94104062ef1")

if(SHERPA_ONNX_USE_STATIC_CRT)
  set(onnxruntime_crt "MT")
else()
  set(onnxruntime_crt "MD")
endif()

message(STATUS "Use MSVC CRT: ${onnxruntime_crt}")

set(onnxruntime_filename "onnxruntime-win-arm64-${onnxruntime_crt}-${CMAKE_BUILD_TYPE}-1.28.2.tar.bz2")
set(onnxruntime_URL  "https://github.com/csukuangfj/onnxruntime-libs/releases/download/v1.28.2/${onnxruntime_filename}")
set(onnxruntime_HASH "${ONNXRUNTIME_HASH_${onnxruntime_crt}_${CMAKE_BUILD_TYPE}}")

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
