function(download_kaldi_hmm_gmm)
  include(FetchContent)

  set(kaldi_hmm_gmm_URL  "https://github.com/csukuangfj/kaldi-hmm-gmm/archive/refs/tags/v1.1.2.tar.gz")
  set(kaldi_hmm_gmm_URL2 "https://huggingface.co/csukuangfj/sherpa-onnx-cmake-deps/resolve/main/kaldi-hmm-gmm-1.1.2.tar.gz")
  set(kaldi_hmm_gmm_HASH "SHA256=7ccf2138139884156bafef979a4c4882f8cd808fa250adcc3699bdaca59cda14")

  set(KHG_ENABLE_TESTS OFF CACHE BOOL "" FORCE)
  set(KHG_BUILD_PYTHON OFF CACHE BOOL "" FORCE)

  # If you don't have access to the Internet,
  # please pre-download kaldi-hmm-gmm
  set(possible_file_locations
    $ENV{HOME}/Downloads/kaldi-hmm-gmm-1.1.2.tar.gz
    ${PROJECT_SOURCE_DIR}/kaldi-hmm-gmm-1.1.2.tar.gz
    ${PROJECT_BINARY_DIR}/kaldi-hmm-gmm-1.1.2.tar.gz
    /tmp/kaldi-hmm-gmm-1.1.2.tar.gz
    /star-fj/fangjun/download/github/kaldi-hmm-gmm-1.1.2.tar.gz
  )

  foreach(f IN LISTS possible_file_locations)
    if(EXISTS ${f})
      set(kaldi_hmm_gmm_URL  "${f}")
      file(TO_CMAKE_PATH "${kaldi_hmm_gmm_URL}" kaldi_hmm_gmm_URL)
      message(STATUS "Found local downloaded kaldi-hmm-gmm: ${kaldi_hmm_gmm_URL}")
      set(kaldi_hmm_gmm_URL2 )
      break()
    endif()
  endforeach()

  FetchContent_Declare(kaldi_hmm_gmm
    URL
      ${kaldi_hmm_gmm_URL}
      ${kaldi_hmm_gmm_URL2}
    URL_HASH          ${kaldi_hmm_gmm_HASH}
  )

  FetchContent_GetProperties(kaldi_hmm_gmm)
  if(NOT kaldi_hmm_gmm_POPULATED)
    message(STATUS "Downloading kaldi-hmm-gmm from ${kaldi_hmm_gmm_URL}")
    FetchContent_Populate(kaldi_hmm_gmm)
  endif()
  message(STATUS "kaldi-hmm-gmm is downloaded to ${kaldi_hmm_gmm_SOURCE_DIR}")
  message(STATUS "kaldi-hmm-gmm's binary dir is ${kaldi_hmm_gmm_BINARY_DIR}")

  include_directories(${kaldi_hmm_gmm_SOURCE_DIR})
  add_subdirectory(${kaldi_hmm_gmm_SOURCE_DIR} ${kaldi_hmm_gmm_BINARY_DIR} EXCLUDE_FROM_ALL)

  target_include_directories(kaldi-hmm-gmm-core
    INTERFACE
      ${kaldi-hmm-gmm_SOURCE_DIR}/
  )
  if(SHERPA_ONNX_ENABLE_PYTHON AND WIN32)
    install(TARGETS kaldi-hmm-gmm-core DESTINATION ..)
  else()
    install(TARGETS kaldi-hmm-gmm-core DESTINATION lib)
  endif()

  if(WIN32 AND BUILD_SHARED_LIBS)
    install(TARGETS kaldi-hmm-gmm-core DESTINATION bin)
  endif()
endfunction()

download_kaldi_hmm_gmm()

