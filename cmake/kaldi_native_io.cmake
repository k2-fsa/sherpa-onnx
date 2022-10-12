function(download_kaldi_native_io)
  if(CMAKE_VERSION VERSION_LESS 3.11)
    # FetchContent is available since 3.11,
    # we've copied it to ${CMAKE_SOURCE_DIR}/cmake/Modules
    # so that it can be used in lower CMake versions.
    message(STATUS "Use FetchContent provided by sherpa-onnx")
    list(APPEND CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/cmake/Modules)
  endif()

  include(FetchContent)

  set(kaldi_native_io_URL  "https://github.com/csukuangfj/kaldi_native_io/archive/refs/tags/v1.15.1.tar.gz")
  set(kaldi_native_io_HASH "SHA256=97377e1d61e99d8fc1d6037a418d3037522dfa46337e06162e24b1d97f3d70a6")

  set(KALDI_NATIVE_IO_BUILD_TESTS OFF CACHE BOOL "" FORCE)
  set(KALDI_NATIVE_IO_BUILD_PYTHON OFF CACHE BOOL "" FORCE)

  FetchContent_Declare(kaldi_native_io
    URL               ${kaldi_native_io_URL}
    URL_HASH          ${kaldi_native_io_HASH}
  )

  FetchContent_GetProperties(kaldi_native_io)
  if(NOT kaldi_native_io_POPULATED)
    message(STATUS "Downloading kaldi_native_io ${kaldi_native_io_URL}")
    FetchContent_Populate(kaldi_native_io)
  endif()
  message(STATUS "kaldi_native_io is downloaded to ${kaldi_native_io_SOURCE_DIR}")
  message(STATUS "kaldi_native_io's binary dir is ${kaldi_native_io_BINARY_DIR}")

  add_subdirectory(${kaldi_native_io_SOURCE_DIR} ${kaldi_native_io_BINARY_DIR} EXCLUDE_FROM_ALL)

  target_include_directories(kaldi_native_io_core
    PUBLIC
      ${kaldi_native_io_SOURCE_DIR}/
  )
endfunction()

download_kaldi_native_io()
