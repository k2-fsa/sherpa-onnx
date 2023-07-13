function(download_onnxruntime)
  include(FetchContent)

  message(STATUS "CMAKE_SYSTEM_NAME: ${CMAKE_SYSTEM_NAME}")
  message(STATUS "CMAKE_SYSTEM_PROCESSOR: ${CMAKE_SYSTEM_PROCESSOR}")

  if(CMAKE_SYSTEM_NAME STREQUAL Linux AND CMAKE_SYSTEM_PROCESSOR STREQUAL aarch64)
    # For embedded systems
    set(possible_file_locations
      $ENV{HOME}/Downloads/onnxruntime-linux-aarch64-1.15.1.tgz
      ${PROJECT_SOURCE_DIR}/onnxruntime-linux-aarch64-1.15.1.tgz
      ${PROJECT_BINARY_DIR}/onnxruntime-linux-aarch64-1.15.1.tgz
      /tmp/onnxruntime-linux-aarch64-1.15.1.tgz
      /star-fj/fangjun/download/github/onnxruntime-linux-aarch64-1.15.1.tgz
    )
    set(onnxruntime_URL  "https://github.com/microsoft/onnxruntime/releases/download/v1.15.1/onnxruntime-linux-aarch64-1.15.1.tgz")
    set(onnxruntime_URL2 "https://huggingface.co/csukuangfj/sherpa-onnx-cmake-deps/resolve/main/onnxruntime-linux-aarch64-1.15.1.tgz")
    set(onnxruntime_HASH "SHA256=85272e75d8dd841138de4b774a9672ea93c1be108d96038c6c34a62d7f976aee")
  elseif(CMAKE_SYSTEM_NAME STREQUAL Linux AND CMAKE_SYSTEM_PROCESSOR STREQUAL arm)
    # For embedded systems
    set(possible_file_locations
      $ENV{HOME}/Downloads/onnxruntime-linux-arm-1.15.1.zip
      ${PROJECT_SOURCE_DIR}/onnxruntime-linux-arm-1.15.1.zip
      ${PROJECT_BINARY_DIR}/onnxruntime-linux-arm-1.15.1.zip
      /tmp/onnxruntime-linux-arm-1.15.1.zip
      /star-fj/fangjun/download/github/onnxruntime-linux-arm-1.15.1.zip
    )
    set(onnxruntime_URL  "https://huggingface.co/csukuangfj/onnxruntime-libs/resolve/main/onnxruntime-linux-arm-1.15.1.zip")
    set(onnxruntime_URL2 "https://huggingface.co/csukuangfj/sherpa-onnx-cmake-deps/resolve/main/onnxruntime-linux-arm-1.15.1.zip")
    set(onnxruntime_HASH "SHA256=867b96210a347e4b1bb949e7c9a3f222371ea0c00c9deaaba9fdd66c689f7fb7")
  elseif(CMAKE_SYSTEM_NAME STREQUAL Linux AND CMAKE_SYSTEM_PROCESSOR STREQUAL x86_64)
    # If you don't have access to the Internet,
    # please pre-download onnxruntime
    set(possible_file_locations
      $ENV{HOME}/Downloads/onnxruntime-linux-x64-1.15.1.tgz
      ${PROJECT_SOURCE_DIR}/onnxruntime-linux-x64-1.15.1.tgz
      ${PROJECT_BINARY_DIR}/onnxruntime-linux-x64-1.15.1.tgz
      /tmp/onnxruntime-linux-x64-1.15.1.tgz
      /star-fj/fangjun/download/github/onnxruntime-linux-x64-1.15.1.tgz
    )

    set(onnxruntime_URL  "https://github.com/microsoft/onnxruntime/releases/download/v1.15.1/onnxruntime-linux-x64-1.15.1.tgz")
    set(onnxruntime_URL2 "https://huggingface.co/csukuangfj/sherpa-onnx-cmake-deps/resolve/main/onnxruntime-linux-x64-1.15.1.tgz")
    set(onnxruntime_HASH "SHA256=5492f9065f87538a286fb04c8542e9ff7950abb2ea6f8c24993a940006787d87")
    # After downloading, it contains:
    #  ./lib/libonnxruntime.so.1.15.1
    #  ./lib/libonnxruntime.so, which is a symlink to lib/libonnxruntime.so.1.15.1
    #
    # ./include
    #    It contains all the needed header files
    if(SHERPA_ONNX_ENABLE_GPU)
      set(onnxruntime_URL "https://github.com/microsoft/onnxruntime/releases/download/v1.15.1/onnxruntime-linux-x64-gpu-1.15.1.tgz")
      set(onnxruntime_URL2 "https://huggingface.co/csukuangfj/sherpa-onnx-cmake-deps/resolve/main/onnxruntime-linux-x64-gpu-1.15.1.tgz")
      set(onnxruntime_HASH "SHA256=eab891393025edd5818d1aa26a42860e5739fcc49e3ca3f876110ec8736fe7f1")

      set(possible_file_locations
        $ENV{HOME}/Downloads/onnxruntime-linux-x64-gpu-1.15.1.tgz
        ${PROJECT_SOURCE_DIR}/onnxruntime-linux-x64-gpu-1.15.1.tgz
        ${PROJECT_BINARY_DIR}/onnxruntime-linux-x64-gpu-1.15.1.tgz
        /tmp/onnxruntime-linux-x64-gpu-1.15.1.tgz
        /star-fj/fangjun/download/github/onnxruntime-linux-x64-gpu-1.15.1.tgz
      )
    endif()
    # After downloading, it contains:
    #  ./lib/libonnxruntime.so.1.15.1
    #  ./lib/libonnxruntime.so, which is a symlink to lib/libonnxruntime.so.1.15.1
    #  ./lib/libonnxruntime_providers_cuda.so
    # ./include, which contains all the needed header files
  elseif(APPLE)
    # If you don't have access to the Internet,
    # please pre-download onnxruntime
    set(possible_file_locations
      $ENV{HOME}/Downloads/onnxruntime-osx-universal2-1.15.1.tgz
      ${PROJECT_SOURCE_DIR}/onnxruntime-osx-universal2-1.15.1.tgz
      ${PROJECT_BINARY_DIR}/onnxruntime-osx-universal2-1.15.1.tgz
      /tmp/onnxruntime-osx-universal2-1.15.1.tgz
    )
    set(onnxruntime_URL  "https://github.com/microsoft/onnxruntime/releases/download/v1.15.1/onnxruntime-osx-universal2-1.15.1.tgz")
    set(onnxruntime_URL2 "https://huggingface.co/csukuangfj/sherpa-onnx-cmake-deps/resolve/main/onnxruntime-osx-universal2-1.15.1.tgz")
    set(onnxruntime_HASH "SHA256=ecb7651c216fe6ffaf4c578e135d98341bc5bc944c5dc6b725ef85b0d7747be0")
    # After downloading, it contains:
    #  ./lib/libonnxruntime.1.15.1.dylib
    #  ./lib/libonnxruntime.dylib, which is a symlink to lib/libonnxruntime.1.15.1.dylib
    #
    # ./include
    #    It contains all the needed header files
  elseif(WIN32)
    message(STATUS "CMAKE_VS_PLATFORM_NAME: ${CMAKE_VS_PLATFORM_NAME}")

    if(CMAKE_VS_PLATFORM_NAME STREQUAL Win32 OR CMAKE_VS_PLATFORM_NAME STREQUAL win32)
      if(BUILD_SHARED_LIBS)
        # If you don't have access to the Internet,
        # please pre-download onnxruntime
        #
        # for 32-bit windows
        set(possible_file_locations
          $ENV{HOME}/Downloads/onnxruntime-win-x86-1.15.1.zip
          ${PROJECT_SOURCE_DIR}/onnxruntime-win-x86-1.15.1.zip
          ${PROJECT_BINARY_DIR}/onnxruntime-win-x86-1.15.1.zip
          /tmp/onnxruntime-win-x86-1.15.1.zip
        )

        set(onnxruntime_URL  "https://github.com/microsoft/onnxruntime/releases/download/v1.15.1/onnxruntime-win-x86-1.15.1.zip")
        set(onnxruntime_URL2 "https://huggingface.co/csukuangfj/sherpa-onnx-cmake-deps/resolve/main/onnxruntime-win-x86-1.15.1.zip")
        set(onnxruntime_HASH "SHA256=8de18fdf274a8adcd95272fcf58beda0fe2fb37f0cd62c02bc4bb6200429e4e2")
      else()
        set(possible_file_locations
          $ENV{HOME}/Downloads/onnxruntime-win-x86-static-1.15.1.tar.bz2
          ${PROJECT_SOURCE_DIR}/onnxruntime-win-x86-static-1.15.1.tar.bz2
          ${PROJECT_BINARY_DIR}/onnxruntime-win-x86-static-1.15.1.tar.bz2
          /tmp/onnxruntime-win-x86-static-1.15.1.tar.bz2
        )

        set(onnxruntime_URL  "https://huggingface.co/csukuangfj/onnxruntime-libs/resolve/main/onnxruntime-win-x86-static-1.15.1.tar.bz2")
        set(onnxruntime_URL2 "")
        set(onnxruntime_HASH "SHA256=a2b33a3e8a1f89cddf303f0a97a5a88f4202579c653cfb29158c8cf7da3734eb")
      endif()

      if(SHERPA_ONNX_ENABLE_GPU)
        message(FATAL_ERROR "GPU support for Win32 is not supported!")
      endif()
    else()
      # for 64-bit windows

      if(BUILD_SHARED_LIBS)
        # If you don't have access to the Internet,
        # please pre-download onnxruntime
        set(possible_file_locations
          $ENV{HOME}/Downloads/onnxruntime-win-x64-1.15.1.zip
          ${PROJECT_SOURCE_DIR}/onnxruntime-win-x64-1.15.1.zip
          ${PROJECT_BINARY_DIR}/onnxruntime-win-x64-1.15.1.zip
          /tmp/onnxruntime-win-x64-1.15.1.zip
        )

        set(onnxruntime_URL  "https://github.com/microsoft/onnxruntime/releases/download/v1.15.1/onnxruntime-win-x64-1.15.1.zip")
        set(onnxruntime_URL2 "https://huggingface.co/csukuangfj/sherpa-onnx-cmake-deps/resolve/main/onnxruntime-win-x64-1.15.1.zip")
        set(onnxruntime_HASH "SHA256=261308ee5526dfd3f405ce8863e43d624a2e0bcd16b2d33cdea8c120ab3534d3")

        if(SHERPA_ONNX_ENABLE_GPU)
          set(onnxruntime_URL  "https://github.com/microsoft/onnxruntime/releases/download/v1.15.1/onnxruntime-win-x64-gpu-1.15.1.zip")
          set(onnxruntime_URL2 "https://huggingface.co/csukuangfj/sherpa-onnx-cmake-deps/resolve/main/onnxruntime-win-x64-gpu-1.15.1.zip")
          set(onnxruntime_HASH "SHA256=dcc3a385b415dd2e4a813018b71da5085d9b97774552edf17947826a255a3732")
          set(possible_file_locations
            $ENV{HOME}/Downloads/onnxruntime-win-x64-gpu-1.15.1.zip
            ${PROJECT_SOURCE_DIR}/onnxruntime-win-x64-gpu-1.15.1.zip
            ${PROJECT_BINARY_DIR}/onnxruntime-win-x64-gpu-1.15.1.zip
            /tmp/onnxruntime-win-x64-gpu-1.15.1.zip
          )
        endif()
      else()
        # static libraries for windows x64
        message(STATUS "Use static onnxruntime libraries")
        # If you don't have access to the Internet,
        # please pre-download onnxruntime
        set(possible_file_locations
          $ENV{HOME}/Downloads/onnxruntime-win-x64-static-1.15.1.tar.bz2
          ${PROJECT_SOURCE_DIR}/onnxruntime-win-x64-static-1.15.1.tar.bz2
          ${PROJECT_BINARY_DIR}/onnxruntime-win-x64-static-1.15.1.tar.bz2
          /tmp/onnxruntime-win-x64-static-1.15.1.tar.bz2
        )

        set(onnxruntime_URL  "https://huggingface.co/csukuangfj/onnxruntime-libs/resolve/main/onnxruntime-win-x64-static-1.15.1.tar.bz2")
        set(onnxruntime_URL2 "")
        set(onnxruntime_HASH "SHA256=f5c19ac1fc6a61c78a231a41df10aede2586665ab397bdc3f007eb8d2c8d4a19")
      endif()
    endif()
    # After downloading, it contains:
    #  ./lib/onnxruntime.{dll,lib,pdb}
    #  ./lib/onnxruntime_providers_shared.{dll,lib,pdb}
    #
    # ./include
    #    It contains all the needed header files
  else()
    message(STATUS "CMAKE_SYSTEM_NAME: ${CMAKE_SYSTEM_NAME}")
    message(STATUS "CMAKE_SYSTEM_PROCESSOR: ${CMAKE_SYSTEM_PROCESSOR}")
    message(FATAL_ERROR "Only support Linux, macOS, and Windows at present. Will support other OSes later")
  endif()

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

  if(BUILD_SHARED_LIBS OR NOT WIN32)
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
  endif()

  if(SHERPA_ONNX_ENABLE_GPU AND NOT WIN32)
    find_library(location_onnxruntime_cuda_lib onnxruntime_providers_cuda
      PATHS
      "${onnxruntime_SOURCE_DIR}/lib"
      NO_CMAKE_SYSTEM_PATH
    )
    add_library(onnxruntime_providers_cuda SHARED IMPORTED)
    set_target_properties(onnxruntime_providers_cuda PROPERTIES
      IMPORTED_LOCATION ${location_onnxruntime_cuda_lib}
    )
  endif()

  if(WIN32)
    if(BUILD_SHARED_LIBS)
      set_property(TARGET onnxruntime
        PROPERTY
          IMPORTED_IMPLIB "${onnxruntime_SOURCE_DIR}/lib/onnxruntime.lib"
      )

      file(COPY ${onnxruntime_SOURCE_DIR}/lib/onnxruntime.dll
        DESTINATION
          ${CMAKE_BINARY_DIR}/bin/${CMAKE_BUILD_TYPE}
      )
      if(SHERPA_ONNX_ENABLE_GPU)
        add_library(onnxruntime_providers_cuda SHARED IMPORTED)

        set_target_properties(onnxruntime_providers_cuda PROPERTIES
          IMPORTED_LOCATION ${location_onnxruntime}
          INTERFACE_INCLUDE_DIRECTORIES "${onnxruntime_SOURCE_DIR}/include"
        )

        set_property(TARGET onnxruntime_providers_cuda
          PROPERTY
            IMPORTED_IMPLIB "${onnxruntime_SOURCE_DIR}/lib/onnxruntime_providers_cuda.lib"
        )

        file(COPY ${onnxruntime_SOURCE_DIR}/lib/onnxruntime_providers_cuda.dll
          DESTINATION
            ${CMAKE_BINARY_DIR}/bin/${CMAKE_BUILD_TYPE}
        )
      endif()
    else()
      # for static libraries, we use onnxruntime_lib_files directly below
      include_directories(${onnxruntime_SOURCE_DIR}/include)
    endif()
  endif()

  if(UNIX AND NOT APPLE)
    file(GLOB onnxruntime_lib_files "${onnxruntime_SOURCE_DIR}/lib/lib*")
  elseif(APPLE)
    file(GLOB onnxruntime_lib_files "${onnxruntime_SOURCE_DIR}/lib/libonnxruntime.*.*dylib")
  elseif(WIN32)
    if(BUILD_SHARED_LIBS)
      file(GLOB onnxruntime_lib_files "${onnxruntime_SOURCE_DIR}/lib/*.dll")
    else()
      file(GLOB onnxruntime_lib_files "${onnxruntime_SOURCE_DIR}/lib/*.lib")
      set(onnxruntime_lib_files ${onnxruntime_lib_files} PARENT_SCOPE)
    endif()
  endif()

  message(STATUS "onnxruntime lib files: ${onnxruntime_lib_files}")
  if(SHERPA_ONNX_ENABLE_PYTHON AND WIN32)
    install(FILES ${onnxruntime_lib_files} DESTINATION ..)
  else()
    install(FILES ${onnxruntime_lib_files} DESTINATION lib)
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
