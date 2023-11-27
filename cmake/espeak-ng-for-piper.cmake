function(download_espeak_ng_for_piper)
  include(FetchContent)

  set(espeak_ng_URL  "https://github.com/csukuangfj/espeak-ng/archive/c97880fa9c1b1c2e0829514afefe72904b40fddb.zip")
  set(espeak_ng_URL2 "")
  set(espeak_ng_HASH "SHA256=fbc678906a9b7d759b9bee72a73ffeca1ff62de58b107284ea60e17c2172e08c")

  set(USE_ASYNC OFF CACHE BOOL "" FORCE)
  set(USE_MBROLA OFF CACHE BOOL "" FORCE)
  set(USE_LIBSONIC OFF CACHE BOOL "" FORCE)
  set(USE_LIBPCAUDIO OFF CACHE BOOL "" FORCE)
  set(USE_KLATT OFF CACHE BOOL "" FORCE)
  set(USE_SPEECHPLAYER OFF CACHE BOOL "" FORCE)
  set(EXTRA_cmn ON CACHE BOOL "" FORCE)
  set(EXTRA_ru ON CACHE BOOL "" FORCE)
  set(CMAKE_C_FLAGS " ${CMAKE_C_FLAGS} -D_FILE_OFFSET_BITS=64 ")

  # If you don't have access to the Internet,
  # please pre-download kaldi-decoder
  set(possible_file_locations
    $ENV{HOME}/Downloads/espeak-ng-c97880fa9c1b1c2e0829514afefe72904b40fddb.zip
    ${PROJECT_SOURCE_DIR}/espeak-ng-c97880fa9c1b1c2e0829514afefe72904b40fddb.zip
    ${PROJECT_BINARY_DIR}/espeak-ng-c97880fa9c1b1c2e0829514afefe72904b40fddb.zip
    /tmp/espeak-ng-c97880fa9c1b1c2e0829514afefe72904b40fddb.zip
    /star-fj/fangjun/download/github/espeak-ng-c97880fa9c1b1c2e0829514afefe72904b40fddb.zip
  )

  foreach(f IN LISTS possible_file_locations)
    if(EXISTS ${f})
      set(espeak_ng_URL  "${f}")
      file(TO_CMAKE_PATH "${espeak_ng_URL}" espeak_ng_URL)
      message(STATUS "Found local downloaded espeak-ng: ${espeak_ng_URL}")
      set(espeak_ng_URL2 )
      break()
    endif()
  endforeach()

  FetchContent_Declare(espeak_ng
    URL
      ${espeak_ng_URL}
      ${espeak_ng_URL2}
    URL_HASH          ${espeak_ng_HASH}
  )

  FetchContent_GetProperties(espeak_ng)
  if(NOT espeak_ng_POPULATED)
    message(STATUS "Downloading espeak-ng from ${espeak_ng_URL}")
    FetchContent_Populate(espeak_ng)
  endif()
  message(STATUS "espeak-ng is downloaded to ${espeak_ng_SOURCE_DIR}")
  message(STATUS "espeak-ng binary dir is ${espeak_ng_BINARY_DIR}")

  add_subdirectory(${espeak_ng_SOURCE_DIR} ${espeak_ng_BINARY_DIR})
  set(espeak_ng_SOURCE_DIR ${espeak_ng_SOURCE_DIR} PARENT_SCOPE)

  target_include_directories(espeak-ng
    INTERFACE
      ${espeak_ng_SOURCE_DIR}/src/include
      ${espeak_ng_SOURCE_DIR}/src/ucd-tools/src/include
  )

  if(SHERPA_ONNX_ENABLE_PYTHON AND WIN32)
    install(TARGETS
      espeak-ng
    DESTINATION ..)
  else()
    install(TARGETS
      espeak-ng
    DESTINATION lib)
  endif()

  if(NOT BUILD_SHARED_LIBS)
    install(TARGETS ucd DESTINATION lib)
  endif()

  if(WIN32 AND BUILD_SHARED_LIBS)
    install(TARGETS
      espeak-ng
    DESTINATION bin)
  endif()
endfunction()

download_espeak_ng_for_piper()
