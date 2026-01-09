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

if(NOT SHERPA_ONNX_ENABLE_GPU)
  message(FATAL_ERROR "This file is for NVIDIA GPU only. Given SHERPA_ONNX_ENABLE_GPU: ${SHERPA_ONNX_ENABLE_GPU}")
endif()

# Requires cuda 12.x, cudnn 9.x
set(onnxruntime_URL  "https://github.com/microsoft/onnxruntime/releases/download/v1.23.2/onnxruntime-win-x64-gpu-1.23.2.zip")
set(onnxruntime_URL2 "https://hf-mirror.com/csukuangfj/onnxruntime-libs/resolve/main/1.23.2/onnxruntime-win-x64-gpu-1.23.2.zip")
set(onnxruntime_HASH "SHA256=e77afdbbc2b8cb6da4e5a50d89841b48c44f3e47dce4fb87b15a2743786d0bb9")

# If you don't have access to the Internet,
# please download onnxruntime to one of the following locations.
# You can add more if you want.
set(possible_file_locations
  $ENV{HOME}/Downloads/onnxruntime-win-x64-gpu-1.23.2.zip
  ${CMAKE_SOURCE_DIR}/onnxruntime-win-x64-gpu-1.23.2.zip
  ${CMAKE_BINARY_DIR}/onnxruntime-win-x64-gpu-1.23.2.zip
  /tmp/onnxruntime-win-x64-gpu-1.23.2.zip
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
