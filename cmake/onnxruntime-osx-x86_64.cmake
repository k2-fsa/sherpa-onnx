# Copyright (c)  2022-2023  Xiaomi Corporation
message(STATUS "CMAKE_SYSTEM_NAME: ${CMAKE_SYSTEM_NAME}")
message(STATUS "CMAKE_SYSTEM_PROCESSOR: ${CMAKE_SYSTEM_PROCESSOR}")
message(STATUS "CMAKE_OSX_ARCHITECTURES: ${CMAKE_OSX_ARCHITECTURES}")
message(STATUS "CMAKE_APPLE_SILICON_PROCESSOR : ${CMAKE_APPLE_SILICON_PROCESSOR}")

if(NOT CMAKE_SYSTEM_NAME STREQUAL Darwin)
  message(FATAL_ERROR "This file is for macOS only. Given: ${CMAKE_SYSTEM_NAME}")
endif()

if(NOT BUILD_SHARED_LIBS)
  message(FATAL_ERROR "This file is for building shared libraries. BUILD_SHARED_LIBS: ${BUILD_SHARED_LIBS}")
endif()

set(onnxruntime_URL  "https://github.com/microsoft/onnxruntime/releases/download/v1.16.3/onnxruntime-osx-x86_64-1.16.3.tgz")
set(onnxruntime_URL2 "https://huggingface.co/csukuangfj/onnxruntime-libs/resolve/main/onnxruntime-osx-x86_64-1.16.3.tgz")
set(onnxruntime_HASH "SHA256=a395923ed91192e46bf50aec48d728dbf193dbd547ad50a231ba32df51608507")

# If you don't have access to the Internet,
# please download onnxruntime to one of the following locations.
# You can add more if you want.
set(possible_file_locations
  $ENV{HOME}/Downloads/onnxruntime-osx-x86_64-1.16.3.tgz
  ${PROJECT_SOURCE_DIR}/onnxruntime-osx-x86_64-1.16.3.tgz
  ${PROJECT_BINARY_DIR}/onnxruntime-osx-x86_64-1.16.3.tgz
  /tmp/onnxruntime-osx-x86_64-1.16.3.tgz
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

file(GLOB onnxruntime_lib_files "${onnxruntime_SOURCE_DIR}/lib/libonnxruntime*dylib")
message(STATUS "onnxruntime lib files: ${onnxruntime_lib_files}")
install(FILES ${onnxruntime_lib_files} DESTINATION lib)
