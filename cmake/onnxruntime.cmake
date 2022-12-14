function(download_onnxruntime)
  if(CMAKE_VERSION VERSION_LESS 3.11)
    # FetchContent is available since 3.11,
    # we've copied it to ${CMAKE_SOURCE_DIR}/cmake/Modules
    # so that it can be used in lower CMake versions.
    message(STATUS "Use FetchContent provided by sherpa-onnx")
    list(APPEND CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/cmake/Modules)
  endif()

  include(FetchContent)

  if(UNIX AND NOT APPLE)
    set(onnxruntime_URL  "https://github.com/microsoft/onnxruntime/releases/download/v1.12.1/onnxruntime-linux-x64-1.12.1.tgz")

    # If you don't have access to the internet, you can first download onnxruntime to some directory, and the use
    # set(onnxruntime_URL  "file:///ceph-fj/fangjun/open-source/sherpa-onnx/onnxruntime-linux-x64-1.12.1.tgz")

    set(onnxruntime_HASH "SHA256=8f6eb9e2da9cf74e7905bf3fc687ef52e34cc566af7af2f92dafe5a5d106aa3d")
    # After downloading, it contains:
    #  ./lib/libonnxruntime.so.1.12.1
    #  ./lib/libonnxruntime.so, which is a symlink to lib/libonnxruntime.so.1.12.1
    #
    # ./include
    #    It contains all the needed header files
  elseif(APPLE)
    set(onnxruntime_URL  "https://github.com/microsoft/onnxruntime/releases/download/v1.12.1/onnxruntime-osx-x86_64-1.12.1.tgz")
    set(onnxruntime_HASH "SHA256=10921c2e75817edcbfc8b29882612be07e25dd33c8449d5892a9d45588898099")
    # After downloading, it contains:
    #  ./lib/libonnxruntime.1.12.1.dylib
    #  ./lib/libonnxruntime.dylib, which is a symlink to lib/libonnxruntime.1.12.1.dylib
    #
    # ./include
    #    It contains all the needed header files
  elseif(WIN32)
    set(onnxruntime_URL  "https://github.com/microsoft/onnxruntime/releases/download/v1.12.1/onnxruntime-win-x64-1.12.1.zip")
    set(onnxruntime_HASH "SHA256=c69650ba14aeae5903b05256a82e77164fff2de992072bc695a3838c1830b85a")
    # After downloading, it contains:
    #  ./lib/onnxruntime.{dll,lib,pdb}
    #  ./lib/onnxruntime_providers_shared.{dll,lib,pdb}
    #
    # ./include
    #    It contains all the needed header files
  else()
    message(FATAL_ERROR "Only support Linux and macOS at present. Will support other OSes later")
  endif()

  FetchContent_Declare(onnxruntime
    URL               ${onnxruntime_URL}
    URL_HASH          ${onnxruntime_HASH}
  )

  FetchContent_GetProperties(onnxruntime)
  if(NOT onnxruntime_POPULATED)
    message(STATUS "Downloading onnxruntime ${onnxruntime_URL}")
    FetchContent_Populate(onnxruntime)
  endif()
  message(STATUS "onnxruntime is downloaded to ${onnxruntime_SOURCE_DIR}")

  find_library(location_onnxruntime onnxruntime
    PATHS
    "${onnxruntime_SOURCE_DIR}/lib"
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
endfunction()

download_onnxruntime()
