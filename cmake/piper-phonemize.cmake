function(download_piper_phonemize)
  include(FetchContent)

  set(piper_phonemize_URL  "https://github.com/csukuangfj/piper-phonemize/archive/8af6256cb3905c785b29305db3ab7ce1e6cd91b2.zip")
  set(piper_phonemize_URL2 "")
  set(piper_phonemize_HASH "SHA256=c0b351389a1680825d3c4ab0d650678c5b8a3c1a92a19368a15ace76f191221f")

  # If you don't have access to the Internet,
  # please pre-download kaldi-decoder
  set(possible_file_locations
    $ENV{HOME}/Downloads/piper-phonemize-8af6256cb3905c785b29305db3ab7ce1e6cd91b2.zip
    ${PROJECT_SOURCE_DIR}/piper-phonemize-8af6256cb3905c785b29305db3ab7ce1e6cd91b2.zip
    ${PROJECT_BINARY_DIR}/piper-phonemize-8af6256cb3905c785b29305db3ab7ce1e6cd91b2.zip
    /tmp/piper-phonemize-8af6256cb3905c785b29305db3ab7ce1e6cd91b2.zip
    /star-fj/fangjun/download/github/piper-phonemize-8af6256cb3905c785b29305db3ab7ce1e6cd91b2.zip
  )

  foreach(f IN LISTS possible_file_locations)
    if(EXISTS ${f})
      set(piper_phonemize_URL  "${f}")
      file(TO_CMAKE_PATH "${piper_phonemize_URL}" piper_phonemize_URL)
      message(STATUS "Found local downloaded espeak-ng: ${piper_phonemize_URL}")
      set(piper_phonemize_URL2 )
      break()
    endif()
  endforeach()

  FetchContent_Declare(piper_phonemize
    URL
      ${piper_phonemize_URL}
      ${piper_phonemize_URL2}
    URL_HASH          ${piper_phonemize_HASH}
  )

  FetchContent_GetProperties(piper_phonemize)
  if(NOT piper_phonemize_POPULATED)
    message(STATUS "Downloading piper-phonemize from ${piper_phonemize_URL}")
    FetchContent_Populate(piper_phonemize)
  endif()
  message(STATUS "piper-phonemize is downloaded to ${piper_phonemize_SOURCE_DIR}")
  message(STATUS "piper-phonemize binary dir is ${piper_phonemize_BINARY_DIR}")

  add_subdirectory(${piper_phonemize_SOURCE_DIR} ${piper_phonemize_BINARY_DIR} EXCLUDE_FROM_ALL)

  if(WIN32 AND MSVC)
    target_compile_options(piper_phonemize PUBLIC
      /wd4309
    )
  endif()

  target_include_directories(piper_phonemize
    INTERFACE
      ${piper_phonemize_SOURCE_DIR}/src/include
  )

  if(SHERPA_ONNX_ENABLE_PYTHON AND WIN32)
    install(TARGETS
      piper_phonemize
    DESTINATION ..)
  else()
    install(TARGETS
      piper_phonemize
    DESTINATION lib)
  endif()

  if(WIN32 AND BUILD_SHARED_LIBS)
    install(TARGETS
      piper_phonemize
    DESTINATION bin)
  endif()
endfunction()

download_piper_phonemize()
