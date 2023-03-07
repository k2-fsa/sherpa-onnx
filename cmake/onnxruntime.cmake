function(download_onnxruntime)
  include(FetchContent)

  if(CMAKE_SYSTEM_PROCESSOR STREQUAL aarch64)
    # For embedded systems
    set(possible_file_locations
      $ENV{HOME}/Downloads/onnxruntime-linux-aarch64-1.14.0.tgz
      ${PROJECT_SOURCE_DIR}/onnxruntime-linux-aarch64-1.14.0.tgz
      ${PROJECT_BINARY_DIR}/onnxruntime-linux-aarch64-1.14.0.tgz
      /tmp/onnxruntime-linux-aarch64-1.14.0.tgz
      /star-fj/fangjun/download/github/onnxruntime-linux-aarch64-1.14.0.tgz
    )
    set(onnxruntime_URL  "https://github.com/microsoft/onnxruntime/releases/download/v1.14.0/onnxruntime-linux-aarch64-1.14.0.tgz")
    set(onnxruntime_URL2 "https://huggingface.co/csukuangfj/sherpa-onnx-cmake-deps/resolve/main/onnxruntime-linux-aarch64-1.14.0.tgz")
    set(onnxruntime_HASH "SHA256=9384d2e6e29fed693a4630303902392eead0c41bee5705ccac6d6d34a3d5db86")
  elseif(CMAKE_SYSTEM_NAME STREQUAL Linux AND CMAKE_SYSTEM_PROCESSOR STREQUAL x86_64)
    # If you don't have access to the Internet,
    # please pre-download onnxruntime
    set(possible_file_locations
      $ENV{HOME}/Downloads/onnxruntime-linux-x64-1.14.0.tgz
      ${PROJECT_SOURCE_DIR}/onnxruntime-linux-x64-1.14.0.tgz
      ${PROJECT_BINARY_DIR}/onnxruntime-linux-x64-1.14.0.tgz
      /tmp/onnxruntime-linux-x64-1.14.0.tgz
      /star-fj/fangjun/download/github/onnxruntime-linux-x64-1.14.0.tgz
    )

    set(onnxruntime_URL  "https://github.com/microsoft/onnxruntime/releases/download/v1.14.0/onnxruntime-linux-x64-1.14.0.tgz")
    set(onnxruntime_URL2 "https://huggingface.co/csukuangfj/sherpa-onnx-cmake-deps/resolve/main/onnxruntime-linux-x64-1.14.0.tgz")
    set(onnxruntime_HASH "SHA256=92bf534e5fa5820c8dffe9de2850f84ed2a1c063e47c659ce09e8c7938aa2090")
    # After downloading, it contains:
    #  ./lib/libonnxruntime.so.1.14.0
    #  ./lib/libonnxruntime.so, which is a symlink to lib/libonnxruntime.so.1.14.0
    #
    # ./include
    #    It contains all the needed header files
  elseif(APPLE)
    # If you don't have access to the Internet,
    # please pre-download onnxruntime
    set(possible_file_locations
      $ENV{HOME}/Downloads/onnxruntime-osx-universal2-1.14.0.tgz
      ${PROJECT_SOURCE_DIR}/onnxruntime-osx-universal2-1.14.0.tgz
      ${PROJECT_BINARY_DIR}/onnxruntime-osx-universal2-1.14.0.tgz
      /tmp/onnxruntime-osx-universal2-1.14.0.tgz
    )
    set(onnxruntime_URL  "https://github.com/microsoft/onnxruntime/releases/download/v1.14.0/onnxruntime-osx-universal2-1.14.0.tgz")
    set(onnxruntime_URL2 "https://huggingface.co/csukuangfj/sherpa-onnx-cmake-deps/resolve/main/onnxruntime-osx-universal2-1.14.0.tgz")
    set(onnxruntime_HASH "SHA256=348563df91f17a2ac010519f37c3b46fd5b79140974e5c5a90a57e032bb25925")
    # After downloading, it contains:
    #  ./lib/libonnxruntime.1.14.0.dylib
    #  ./lib/libonnxruntime.dylib, which is a symlink to lib/libonnxruntime.1.14.0.dylib
    #
    # ./include
    #    It contains all the needed header files
  elseif(WIN32)
    message(STATUS "CMAKE_VS_PLATFORM_NAME: ${CMAKE_VS_PLATFORM_NAME}")

    if(CMAKE_VS_PLATFORM_NAME STREQUAL Win32)
      # If you don't have access to the Internet,
      # please pre-download onnxruntime
      #
      # for 32-bit windows
      set(possible_file_locations
        $ENV{HOME}/Downloads/onnxruntime-win-x86-1.14.0.zip
        ${PROJECT_SOURCE_DIR}/onnxruntime-win-x86-1.14.0.zip
        ${PROJECT_BINARY_DIR}/onnxruntime-win-x86-1.14.0.zip
        /tmp/onnxruntime-win-x86-1.14.0.zip
      )

      set(onnxruntime_URL  "https://github.com/microsoft/onnxruntime/releases/download/v1.14.0/onnxruntime-win-x86-1.14.0.zip")
      set(onnxruntime_URL2 "https://huggingface.co/csukuangfj/sherpa-onnx-cmake-deps/resolve/main/onnxruntime-win-x86-1.14.0.zip")
      set(onnxruntime_HASH "SHA256=4214b130db602cbf31a6f26f25377ab077af0cf03c4ddd4651283e1fb68f56cf")
    else()
      # If you don't have access to the Internet,
      # please pre-download onnxruntime
      #
      # for 64-bit windows
      set(possible_file_locations
        $ENV{HOME}/Downloads/onnxruntime-win-x64-1.14.0.zip
        ${PROJECT_SOURCE_DIR}/onnxruntime-win-x64-1.14.0.zip
        ${PROJECT_BINARY_DIR}/onnxruntime-win-x64-1.14.0.zip
        /tmp/onnxruntime-win-x64-1.14.0.zip
      )

      set(onnxruntime_URL  "https://github.com/microsoft/onnxruntime/releases/download/v1.14.0/onnxruntime-win-x64-1.14.0.zip")
      set(onnxruntime_URL2 "https://huggingface.co/csukuangfj/sherpa-onnx-cmake-deps/resolve/main/onnxruntime-win-x64-1.14.0.zip")
      set(onnxruntime_HASH "SHA256=300eafef456748cde2743ee08845bd40ff1bab723697ff934eba6d4ce3519620")
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
  if(WIN32)
    set_property(TARGET onnxruntime
      PROPERTY
        IMPORTED_IMPLIB "${onnxruntime_SOURCE_DIR}/lib/onnxruntime.lib"
    )

    file(COPY ${onnxruntime_SOURCE_DIR}/lib/onnxruntime.dll
      DESTINATION
        ${CMAKE_BINARY_DIR}/bin/${CMAKE_BUILD_TYPE}
    )
  endif()

  if(UNIX AND NOT APPLE)
    file(GLOB onnxruntime_lib_files "${onnxruntime_SOURCE_DIR}/lib/lib*")
  elseif(APPLE)
    file(GLOB onnxruntime_lib_files "${onnxruntime_SOURCE_DIR}/lib/lib*dylib")
  elseif(WIN32)
    file(GLOB onnxruntime_lib_files "${onnxruntime_SOURCE_DIR}/lib/*.dll")
  endif()

  message(STATUS "onnxruntime lib files: ${onnxruntime_lib_files}")
  install(FILES ${onnxruntime_lib_files} DESTINATION lib)
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
  set(location_onnxruntime_lib $ENV{SHERPA_ONNXRUNTIME_LIB_DIR}/libonnxruntime.so)
  if(NOT EXISTS ${location_onnxruntime_lib})
    set(location_onnxruntime_lib $ENV{SHERPA_ONNXRUNTIME_LIB_DIR}/libonnxruntime.a)
  endif()
else()
  find_library(location_onnxruntime_lib onnxruntime
    PATHS
      /lib
      /usr/lib
      /usr/local/lib
  )
endif()

message(STATUS "location_onnxruntime_lib: ${location_onnxruntime_lib}")

if(location_onnxruntime_header_dir AND location_onnxruntime_lib)
  add_library(onnxruntime SHARED IMPORTED)
  set_target_properties(onnxruntime PROPERTIES
    IMPORTED_LOCATION ${location_onnxruntime_lib}
    INTERFACE_INCLUDE_DIRECTORIES "${location_onnxruntime_header_dir}"
  )
else()
  message(STATUS "Could not find a pre-installed onnxruntime. Downloading pre-compiled onnxruntime")
  download_onnxruntime()
endif()
