# Copyright (c)  2022-2023  Xiaomi Corporation
function(download_onnxruntime)
  include(FetchContent)

  message(STATUS "CMAKE_SYSTEM_NAME: ${CMAKE_SYSTEM_NAME}")
  message(STATUS "CMAKE_SYSTEM_PROCESSOR: ${CMAKE_SYSTEM_PROCESSOR}")

  if(CMAKE_SYSTEM_NAME STREQUAL Linux AND CMAKE_SYSTEM_PROCESSOR STREQUAL aarch64)
    include(onnxruntime-linux-aarch64)
  elseif(CMAKE_SYSTEM_NAME STREQUAL Linux AND CMAKE_SYSTEM_PROCESSOR STREQUAL arm)
    include(onnxruntime-linux-arm)
  elseif(CMAKE_SYSTEM_NAME STREQUAL Linux AND CMAKE_SYSTEM_PROCESSOR STREQUAL x86_64)
    if(SHERPA_ONNX_ENABLE_GPU)
      include(onnxruntime-linux-x86_64-gpu)
    else()
      include(onnxruntime-linux-x86_64)
    endif()
  elseif(CMAKE_SYSTEM_NAME STREQUAL Darwin)
    if (arm64 IN_LIST CMAKE_OSX_ARCHITECTURES AND x86_64 IN_LIST CMAKE_OSX_ARCHITECTURES)
      include(onnxruntime-osx-universal)
    elseif(CMAKE_SYSTEM_PROCESSOR STREQUAL x86_64 AND CMAKE_OSX_ARCHITECTURES STREQUAL "arm64")
      # cross compiling
      include(onnxruntime-osx-arm64)
    elseif(CMAKE_SYSTEM_PROCESSOR STREQUAL arm64 AND CMAKE_OSX_ARCHITECTURES STREQUAL "x86_64")
      # cross compiling
      include(onnxruntime-osx-x86_64)
    elseif(CMAKE_SYSTEM_PROCESSOR STREQUAL arm64)
      include(onnxruntime-osx-arm64)
    elseif(CMAKE_SYSTEM_PROCESSOR STREQUAL x86_64)
      include(onnxruntime-osx-x86_64)
    else()
      message(FATAL_ERROR "Unsupport processor {CMAKE_SYSTEM_PROCESSOR} for Darwin")
    endif()
  elseif(WIN32)
    message(STATUS "CMAKE_VS_PLATFORM_NAME: ${CMAKE_VS_PLATFORM_NAME}")

    if(CMAKE_VS_PLATFORM_NAME STREQUAL Win32 OR CMAKE_VS_PLATFORM_NAME STREQUAL win32)
      if(BUILD_SHARED_LIBS)
        include(onnxruntime-win-x86)
      else()
        include(onnxruntime-win-x86-static)
      endif()

      if(SHERPA_ONNX_ENABLE_GPU)
        message(FATAL_ERROR "GPU support for Win32 is not supported!")
      endif()
    else()
      # for 64-bit windows

      if(BUILD_SHARED_LIBS)
        if(SHERPA_ONNX_ENABLE_GPU)
          include(onnxruntime-win-x64-gpu)
        else()
          include(onnxruntime-win-x64)
        endif()
      else()
        # static libraries for windows x64
        message(STATUS "Use static onnxruntime libraries")
        include(onnxruntime-win-x64-static)
      endif()
    endif()
  else()
    message(STATUS "CMAKE_SYSTEM_NAME: ${CMAKE_SYSTEM_NAME}")
    message(STATUS "CMAKE_SYSTEM_PROCESSOR: ${CMAKE_SYSTEM_PROCESSOR}")
    message(FATAL_ERROR "Only support Linux, macOS, and Windows at present. Will support other OSes later")
  endif()
endfunction()

# First, we try to locate the header and the lib if the use has already
# installed onnxruntime. Otherwise, we will download the pre-compiled lib

message(STATUS "CMAKE_SYSTEM_NAME: ${CMAKE_SYSTEM_NAME}")
message(STATUS "CMAKE_SYSTEM_PROCESSOR: ${CMAKE_SYSTEM_PROCESSOR}")

if(DEFINED ENV{SHERPA_ONNXRUNTIME_INCLUDE_DIR})
  set(location_onnxruntime_header_dir $ENV{SHERPA_ONNXRUNTIME_INCLUDE_DIR})
else()
  find_path(location_onnxruntime_header_dir onnxruntime_cxx_api.h
    PATHS
      /usr/include
      /usr/local/include
  )
endif()

message(STATUS "location_onnxruntime_header_dir: ${location_onnxruntime_header_dir}")

if(DEFINED ENV{SHERPA_ONNXRUNTIME_LIB_DIR})
  if(APPLE)
    set(location_onnxruntime_lib $ENV{SHERPA_ONNXRUNTIME_LIB_DIR}/libonnxruntime.dylib)
  else()
    set(location_onnxruntime_lib $ENV{SHERPA_ONNXRUNTIME_LIB_DIR}/libonnxruntime.so)
  endif()
  if(NOT EXISTS ${location_onnxruntime_lib})
    set(location_onnxruntime_lib $ENV{SHERPA_ONNXRUNTIME_LIB_DIR}/libonnxruntime.a)
  endif()
  if(SHERPA_ONNX_ENABLE_GPU)
    set(location_onnxruntime_cuda_lib $ENV{SHERPA_ONNXRUNTIME_LIB_DIR}/libonnxruntime_providers_cuda.so)
    if(NOT EXISTS ${location_onnxruntime_cuda_lib})
      set(location_onnxruntime_cuda_lib $ENV{SHERPA_ONNXRUNTIME_LIB_DIR}/libonnxruntime_providers_cuda.a)
    endif()
  endif()
else()
  find_library(location_onnxruntime_lib onnxruntime
    PATHS
      /lib
      /usr/lib
      /usr/local/lib
  )

  if(SHERPA_ONNX_ENABLE_GPU)
    find_library(location_onnxruntime_cuda_lib onnxruntime_providers_cuda
      PATHS
        /lib
        /usr/lib
        /usr/local/lib
    )
  endif()
endif()

message(STATUS "location_onnxruntime_lib: ${location_onnxruntime_lib}")
if(SHERPA_ONNX_ENABLE_GPU)
  message(STATUS "location_onnxruntime_cuda_lib: ${location_onnxruntime_cuda_lib}")
endif()

if(location_onnxruntime_header_dir AND location_onnxruntime_lib)
  add_library(onnxruntime SHARED IMPORTED)
  set_target_properties(onnxruntime PROPERTIES
    IMPORTED_LOCATION ${location_onnxruntime_lib}
    INTERFACE_INCLUDE_DIRECTORIES "${location_onnxruntime_header_dir}"
  )
  if(SHERPA_ONNX_ENABLE_GPU AND location_onnxruntime_cuda_lib)
    add_library(onnxruntime_providers_cuda SHARED IMPORTED)
    set_target_properties(onnxruntime_providers_cuda PROPERTIES
      IMPORTED_LOCATION ${location_onnxruntime_cuda_lib}
    )
  endif()
else()
  message(STATUS "Could not find a pre-installed onnxruntime. Downloading pre-compiled onnxruntime")
  download_onnxruntime()
endif()
