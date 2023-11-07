function(download_kaldi_decoder)
  include(FetchContent)

  set(kaldi_decoder_URL  "https://github.com/k2-fsa/kaldi-decoder/archive/refs/tags/v0.2.3.tar.gz")
  set(kaldi_decoder_URL2 "https://huggingface.co/csukuangfj/sherpa-onnx-cmake-deps/resolve/main/kaldi-decoder-0.2.3.tar.gz")
  set(kaldi_decoder_HASH "SHA256=98bf445a5b7961ccf3c3522317d900054eaadb6a9cdcf4531e7d9caece94a56d")

  set(KALDI_DECODER_BUILD_PYTHON OFF CACHE BOOL "" FORCE)
  set(KALDI_DECODER_ENABLE_TESTS OFF CACHE BOOL "" FORCE)
  set(KALDIFST_BUILD_PYTHON OFF CACHE BOOL "" FORCE)

  # If you don't have access to the Internet,
  # please pre-download kaldi-decoder
  set(possible_file_locations
    $ENV{HOME}/Downloads/kaldi-decoder-0.2.3.tar.gz
    ${PROJECT_SOURCE_DIR}/kaldi-decoder-0.2.3.tar.gz
    ${PROJECT_BINARY_DIR}/kaldi-decoder-0.2.3.tar.gz
    /tmp/kaldi-decoder-0.2.3.tar.gz
    /star-fj/fangjun/download/github/kaldi-decoder-0.2.3.tar.gz
  )

  foreach(f IN LISTS possible_file_locations)
    if(EXISTS ${f})
      set(kaldi_decoder_URL  "${f}")
      file(TO_CMAKE_PATH "${kaldi_decoder_URL}" kaldi_decoder_URL)
      message(STATUS "Found local downloaded kaldi-decoder: ${kaldi_decoder_URL}")
      set(kaldi_decoder_URL2 )
      break()
    endif()
  endforeach()

  FetchContent_Declare(kaldi_decoder
    URL
      ${kaldi_decoder_URL}
      ${kaldi_decoder_URL2}
    URL_HASH          ${kaldi_decoder_HASH}
  )

  FetchContent_GetProperties(kaldi_decoder)
  if(NOT kaldi_decoder_POPULATED)
    message(STATUS "Downloading kaldi-decoder from ${kaldi_decoder_URL}")
    FetchContent_Populate(kaldi_decoder)
  endif()
  message(STATUS "kaldi-decoder is downloaded to ${kaldi_decoder_SOURCE_DIR}")
  message(STATUS "kaldi-decoder's binary dir is ${kaldi_decoder_BINARY_DIR}")

  include_directories(${kaldi_decoder_SOURCE_DIR})
  add_subdirectory(${kaldi_decoder_SOURCE_DIR} ${kaldi_decoder_BINARY_DIR} EXCLUDE_FROM_ALL)

  target_include_directories(kaldi-decoder-core
    INTERFACE
      ${kaldi-decoder_SOURCE_DIR}/
  )
  if(SHERPA_ONNX_ENABLE_PYTHON AND WIN32)
    install(TARGETS
      kaldi-decoder-core
      kaldifst_core
      fst
    DESTINATION ..)
  else()
    install(TARGETS
      kaldi-decoder-core
      kaldifst_core
      fst
    DESTINATION lib)
  endif()

  if(WIN32 AND BUILD_SHARED_LIBS)
    install(TARGETS
      kaldi-decoder-core
      kaldifst_core
      fst
    DESTINATION bin)
  endif()
endfunction()

download_kaldi_decoder()

