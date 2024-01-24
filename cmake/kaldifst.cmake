function(download_kaldifst)
  include(FetchContent)

  set(kaldifst_URL  "https://github.com/k2-fsa/kaldifst/archive/refs/tags/v1.7.9.tar.gz")
  set(kaldifst_URL2 "https://huggingface.co/csukuangfj/kaldi-hmm-gmm-cmake-deps/resolve/main/kaldifst-1.7.9.tar.gz")
  set(kaldifst_HASH "SHA256=8c653021491dca54c38ab659565edfab391418a79ae87099257863cd5664dd39")

  # If you don't have access to the Internet,
  # please pre-download kaldifst
  set(possible_file_locations
    $ENV{HOME}/Downloads/kaldifst-1.7.9.tar.gz
    ${CMAKE_SOURCE_DIR}/kaldifst-1.7.9.tar.gz
    ${CMAKE_BINARY_DIR}/kaldifst-1.7.9.tar.gz
    /tmp/kaldifst-1.7.9.tar.gz
    /star-fj/fangjun/download/github/kaldifst-1.7.9.tar.gz
  )

  foreach(f IN LISTS possible_file_locations)
    if(EXISTS ${f})
      set(kaldifst_URL  "${f}")
      file(TO_CMAKE_PATH "${kaldifst_URL}" kaldifst_URL)
      message(STATUS "Found local downloaded kaldifst: ${kaldifst_URL}")
      set(kaldifst_URL2)
      break()
    endif()
  endforeach()

  set(KALDIFST_BUILD_TESTS OFF CACHE BOOL "" FORCE)
  set(KALDIFST_BUILD_PYTHON OFF CACHE BOOL "" FORCE)

  FetchContent_Declare(kaldifst
    URL               ${kaldifst_URL}
    URL_HASH          ${kaldifst_HASH}
  )

  FetchContent_GetProperties(kaldifst)
  if(NOT kaldifst_POPULATED)
    message(STATUS "Downloading kaldifst from ${kaldifst_URL}")
    FetchContent_Populate(kaldifst)
  endif()
  message(STATUS "kaldifst is downloaded to ${kaldifst_SOURCE_DIR}")
  message(STATUS "kaldifst's binary dir is ${kaldifst_BINARY_DIR}")

  list(APPEND CMAKE_MODULE_PATH ${kaldifst_SOURCE_DIR}/cmake)

  add_subdirectory(${kaldifst_SOURCE_DIR} ${kaldifst_BINARY_DIR} EXCLUDE_FROM_ALL)

  target_include_directories(kaldifst_core
    PUBLIC
      ${kaldifst_SOURCE_DIR}/
  )

  target_include_directories(fst
    PUBLIC
      ${openfst_SOURCE_DIR}/src/include
  )

  set_target_properties(kaldifst_core PROPERTIES OUTPUT_NAME "sherpa-onnx-kaldifst-core")
  set_target_properties(fst PROPERTIES OUTPUT_NAME "sherpa-onnx-fst")
endfunction()

download_kaldifst()
