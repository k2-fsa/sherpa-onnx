function(download_kaldi_native_fbank)
  include(FetchContent)

  set(kaldi_native_fbank_URL  "https://github.com/csukuangfj/kaldi-native-fbank/archive/refs/tags/v1.12.tar.gz")
  set(kaldi_native_fbank_HASH "SHA256=8f4dfc3f6ddb1adcd9ac0ae87743ebc6cbcae147aacf9d46e76fa54134e12b44")

  set(KALDI_NATIVE_FBANK_BUILD_TESTS OFF CACHE BOOL "" FORCE)
  set(KALDI_NATIVE_FBANK_BUILD_PYTHON OFF CACHE BOOL "" FORCE)
  set(KALDI_NATIVE_FBANK_ENABLE_CHECK OFF CACHE BOOL "" FORCE)

  # If you don't have access to the Internet,
  # please pre-download kaldi-native-fbank
  set(possible_file_locations
    $ENV{HOME}/Downloads/kaldi-native-fbank-1.12.tar.gz
    ${PROJECT_SOURCE_DIR}/kaldi-native-fbank-1.12.tar.gz
    ${PROJECT_BINARY_DIR}/kaldi-native-fbank-1.12.tar.gz
    /tmp/kaldi-native-fbank-1.12.tar.gz
    /star-fj/fangjun/download/github/kaldi-native-fbank-1.12.tar.gz
  )

  foreach(f IN LISTS possible_file_locations)
    if(EXISTS ${f})
      set(kaldi_native_fbank_URL  "file://${f}")
      break()
    endif()
  endforeach()

  FetchContent_Declare(kaldi_native_fbank
    URL               ${kaldi_native_fbank_URL}
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
endfunction()

download_kaldi_native_fbank()
