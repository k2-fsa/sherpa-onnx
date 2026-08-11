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

if(NOT BUILD_SHARED_LIBS)
  message(FATAL_ERROR "This file is for building shared libraries. BUILD_SHARED_LIBS: ${BUILD_SHARED_LIBS}")
endif()

if(NOT CMAKE_BUILD_TYPE MATCHES "^(Release|Debug|RelWithDebInfo|MinSizeRel)$")
  message(FATAL_ERROR "Supported CMAKE_BUILD_TYPE values are: Release, Debug, RelWithDebInfo, MinSizeRel. Given ${CMAKE_BUILD_TYPE}")
endif()

# Hashes for static CRT (/MT)
set(ONNXRUNTIME_HASH_MT_Debug "SHA256=ef2940965429b1bf7d1aedd9bf1dca724deafcab66354ab7dbdb1786a29be0e8")
set(ONNXRUNTIME_HASH_MT_RelWithDebInfo "SHA256=d7ea43568a728c5d1826957092a5ce4157f687253f0c1bbf93f4b5883a310b0c")
set(ONNXRUNTIME_HASH_MT_MinSizeRel "SHA256=f8b626df5443ee2fcbabec35602455a704daa6eae13ef12467e6cff41cd9b6bb")
set(ONNXRUNTIME_HASH_MT_Release "SHA256=25122f517cb114f74009cef7482468f8537eceb7153431f1a2755d062eea0bd7")

# Hashes for dynamic CRT (/MD)
set(ONNXRUNTIME_HASH_MD_Debug "SHA256=fd07dfb9c0433017e4494540d2e511a38416ef020dc9e1865985586dbb44a1d4")
set(ONNXRUNTIME_HASH_MD_RelWithDebInfo "SHA256=ec6c1afef6475d84971858eaad5e226d6febdcabb904d92cd68c1ca2affc9d84")
set(ONNXRUNTIME_HASH_MD_MinSizeRel "SHA256=c99999d5436ecce34ac1e421e2e35b643bcf3db6e2657535f3754809a82a080c")
set(ONNXRUNTIME_HASH_MD_Release "SHA256=cc8ce02c2da575901873731f17b6ab6cbb31a9ce1d83550bccb0a9aac581fec3")

if(SHERPA_ONNX_USE_STATIC_CRT)
  set(onnxruntime_crt "MT")
else()
  set(onnxruntime_crt "MD")
endif()

message(STATUS "Use MSVC CRT: ${onnxruntime_crt}")

set(onnxruntime_filename "onnxruntime-win-x64-${onnxruntime_crt}-${CMAKE_BUILD_TYPE}-1.27.1.tar.bz2")
set(onnxruntime_URL  "https://github.com/csukuangfj/onnxruntime-libs/releases/download/v1.27.1/${onnxruntime_filename}")
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
