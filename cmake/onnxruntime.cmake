function(download_onnxruntime)
  include(FetchContent)

  if(UNIX AND NOT APPLE)
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
      $ENV{HOME}/Downloads/onnxruntime-osx-x86_64-1.14.0.tgz
      ${PROJECT_SOURCE_DIR}/onnxruntime-osx-x86_64-1.14.0.tgz
      ${PROJECT_BINARY_DIR}/onnxruntime-osx-x86_64-1.14.0.tgz
      /tmp/onnxruntime-osx-x86_64-1.14.0.tgz
    )
    set(onnxruntime_URL  "https://github.com/microsoft/onnxruntime/releases/download/v1.14.0/onnxruntime-osx-x86_64-1.14.0.tgz")
    set(onnxruntime_HASH "SHA256=abd2056d56051e78263af37b8dffc3e6da110d2937af7a581a34d1a58dc58360")
    # After downloading, it contains:
    #  ./lib/libonnxruntime.1.14.0.dylib
    #  ./lib/libonnxruntime.dylib, which is a symlink to lib/libonnxruntime.1.14.0.dylib
    #
    # ./include
    #    It contains all the needed header files
  elseif(WIN32)
    # If you don't have access to the Internet,
    # please pre-download onnxruntime
    set(possible_file_locations
      $ENV{HOME}/Downloads/onnxruntime-win-x64-1.14.0.zip
      ${PROJECT_SOURCE_DIR}/onnxruntime-win-x64-1.14.0.zip
      ${PROJECT_BINARY_DIR}/onnxruntime-win-x64-1.14.0.zip
      /tmp/onnxruntime-win-x64-1.14.0.zip
    )
    set(onnxruntime_URL  "https://github.com/microsoft/onnxruntime/releases/download/v1.14.0/onnxruntime-win-x64-1.14.0.zip")
    set(onnxruntime_HASH "SHA256=300eafef456748cde2743ee08845bd40ff1bab723697ff934eba6d4ce3519620")
    # After downloading, it contains:
    #  ./lib/onnxruntime.{dll,lib,pdb}
    #  ./lib/onnxruntime_providers_shared.{dll,lib,pdb}
    #
    # ./include
    #    It contains all the needed header files
  else()
    message(FATAL_ERROR "Only support Linux and macOS at present. Will support other OSes later")
  endif()

  foreach(f IN LISTS possible_file_locations)
    if(EXISTS ${f})
      set(onnxruntime_URL  "file://${f}")
      break()
    endif()
  endforeach()

  FetchContent_Declare(onnxruntime
    URL               ${onnxruntime_URL}
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
