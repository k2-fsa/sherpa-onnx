function(download_utfcpp)
  include(FetchContent)

  set(utfcpp_URL  "https://github.com/nemtrif/utfcpp/archive/refs/tags/v3.2.5.tar.gz")
  set(utfcpp_URL2 "https://huggingface.co/csukuangfj/sherpa-onnx-cmake-deps/resolve/main/utfcpp-3.2.5.tar.gz")
  set(utfcpp_HASH "SHA256=14fd1b3c466814cb4c40771b7f207b61d2c7a0aa6a5e620ca05c00df27f25afd")

  # If you don't have access to the Internet,
  # please pre-download utfcpp
  set(possible_file_locations
    $ENV{HOME}/Downloads/utfcpp-3.2.5.tar.gz
    ${PROJECT_SOURCE_DIR}/utfcpp-3.2.5.tar.gz
    ${PROJECT_BINARY_DIR}/utfcpp-3.2.5.tar.gz
    /tmp/utfcpp-3.2.5.tar.gz
    /star-fj/fangjun/download/github/utfcpp-3.2.5.tar.gz
  )

  foreach(f IN LISTS possible_file_locations)
    if(EXISTS ${f})
      set(utfcpp_URL  "${f}")
      file(TO_CMAKE_PATH "${utfcpp_URL}" utfcpp_URL)
      message(STATUS "Found local downloaded utfcpp: ${utfcpp_URL}")
      set(utfcpp_URL2)
      break()
    endif()
  endforeach()

  FetchContent_Declare(utfcpp
    URL
      ${utfcpp_URL}
      ${utfcpp_URL2}
    URL_HASH          ${utfcpp_HASH}
  )

  FetchContent_GetProperties(utfcpp)
  if(NOT utfcpp_POPULATED)
    message(STATUS "Downloading utfcpp from ${utfcpp_URL}")
    FetchContent_Populate(utfcpp)
  endif()
  message(STATUS "utfcpp is downloaded to ${utfcpp_SOURCE_DIR}")
  # add_subdirectory(${utfcpp_SOURCE_DIR} ${utfcpp_BINARY_DIR} EXCLUDE_FROM_ALL)
  include_directories(${utfcpp_SOURCE_DIR})
endfunction()

download_utfcpp()
