function(download_kaldi_native_fbank)
  include(FetchContent)

  set(kaldi_native_fbank_URL   "https://github.com/csukuangfj/kaldi-native-fbank/archive/refs/tags/v1.19.1.tar.gz")
  set(kaldi_native_fbank_URL2  "https://hub.nuaa.cf/csukuangfj/kaldi-native-fbank/archive/refs/tags/v1.19.1.tar.gz")
  set(kaldi_native_fbank_HASH "SHA256=0cae8cbb9ea42916b214e088912f9e8f2f648f54756b305f93f552382f31f904")

  set(KALDI_NATIVE_FBANK_BUILD_TESTS OFF CACHE BOOL "" FORCE)
  set(KALDI_NATIVE_FBANK_BUILD_PYTHON OFF CACHE BOOL "" FORCE)
  set(KALDI_NATIVE_FBANK_ENABLE_CHECK OFF CACHE BOOL "" FORCE)

  # If you don't have access to the Internet,
  # please pre-download kaldi-native-fbank
  set(possible_file_locations
    $ENV{HOME}/Downloads/kaldi-native-fbank-1.19.1.tar.gz
    ${CMAKE_SOURCE_DIR}/kaldi-native-fbank-1.19.1.tar.gz
    ${CMAKE_BINARY_DIR}/kaldi-native-fbank-1.19.1.tar.gz
    /tmp/kaldi-native-fbank-1.19.1.tar.gz
    /star-fj/fangjun/download/github/kaldi-native-fbank-1.19.1.tar.gz
  )

  foreach(f IN LISTS possible_file_locations)
    if(EXISTS ${f})
      set(kaldi_native_fbank_URL  "${f}")
      file(TO_CMAKE_PATH "${kaldi_native_fbank_URL}" kaldi_native_fbank_URL)
      message(STATUS "Found local downloaded kaldi-native-fbank: ${kaldi_native_fbank_URL}")
      set(kaldi_native_fbank_URL2 )
      break()
    endif()
  endforeach()

  FetchContent_Declare(kaldi_native_fbank
    URL
      ${kaldi_native_fbank_URL}
      ${kaldi_native_fbank_URL2}
    URL_HASH          ${kaldi_native_fbank_HASH}
  )

  FetchContent_GetProperties(kaldi_native_fbank)
  if(NOT kaldi_native_fbank_POPULATED)
    message(STATUS "Downloading kaldi-native-fbank from ${kaldi_native_fbank_URL}")
    FetchContent_Populate(kaldi_native_fbank)
  endif()
  message(STATUS "kaldi-native-fbank is downloaded to ${kaldi_native_fbank_SOURCE_DIR}")
  message(STATUS "kaldi-native-fbank's binary dir is ${kaldi_native_fbank_BINARY_DIR}")

  add_subdirectory(${kaldi_native_fbank_SOURCE_DIR} ${kaldi_native_fbank_BINARY_DIR} EXCLUDE_FROM_ALL)

  target_include_directories(kaldi-native-fbank-core
    INTERFACE
      ${kaldi_native_fbank_SOURCE_DIR}/
  )
  if(SHERPA_ONNX_ENABLE_PYTHON AND WIN32)
    install(TARGETS kaldi-native-fbank-core DESTINATION ..)
  else()
    install(TARGETS kaldi-native-fbank-core DESTINATION lib)
  endif()

  if(WIN32 AND BUILD_SHARED_LIBS)
    install(TARGETS kaldi-native-fbank-core DESTINATION bin)
  endif()
endfunction()

download_kaldi_native_fbank()
