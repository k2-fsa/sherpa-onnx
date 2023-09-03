# Copyright (c)  2022-2023  Xiaomi Corporation

function(download_sentencepiece)
  include(FetchContent)

  set(sentencepiece_URL "https://github.com/google/sentencepiece/archive/refs/tags/v0.1.96.tar.gz")
  set(sentencepiece_HASH "SHA256=5198f31c3bb25e685e9e68355a3bf67a1db23c9e8bdccc33dc015f496a44df7a")

  set(SPM_ENABLE_SHARED ON CACHE BOOL "" FORCE)
  set(SPM_ENABLE_TCMALLOC OFF CACHE BOOL "" FORCE)
  set(SPM_ENABLE_MSVC_MT_BUILD ON CACHE BOOL "" FORCE)

  FetchContent_Declare(sentencepiece
    URL               ${sentencepiece_URL}
    URL_HASH          ${sentencepiece_HASH}
  )

  FetchContent_GetProperties(sentencepiece)
  if(NOT sentencepiece_POPULATED)
    message(STATUS "Downloading sentencepiece")
    FetchContent_Populate(sentencepiece)
  endif()
  message(STATUS "sentencepiece is downloaded to ${sentencepiece_SOURCE_DIR}")
  message(STATUS "sentencepiece's binary dir is ${sentencepiece_BINARY_DIR}")

  add_subdirectory(${sentencepiece_SOURCE_DIR} ${sentencepiece_BINARY_DIR} EXCLUDE_FROM_ALL)

  target_include_directories(sentencepiece
    INTERFACE
      ${sentencepiece_SOURCE_DIR}
      ${sentencepiece_SOURCE_DIR}/src
      ${sentencepiece_BINARY_DIR}
  )

  target_include_directories(sentencepiece-static
    INTERFACE
      ${sentencepiece_SOURCE_DIR}
      ${sentencepiece_SOURCE_DIR}/src
      ${sentencepiece_BINARY_DIR}
  )
endfunction()

download_sentencepiece()
