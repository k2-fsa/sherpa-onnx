function(download_json)
  include(FetchContent)

  set(json_URL  "https://github.com/nlohmann/json/archive/refs/tags/v3.11.2.tar.gz")
  set(json_URL2 "https://huggingface.co/csukuangfj/sherpa-cmake-deps/resolve/main/json-3.11.2.tar.gz")
  set(json_HASH "SHA256=d69f9deb6a75e2580465c6c4c5111b89c4dc2fa94e3a85fcd2ffcd9a143d9273")

  # If you don't have access to the Internet,
  # please pre-download json
  set(possible_file_locations
    $ENV{HOME}/Downloads/json-3.11.2.tar.gz
    ${PROJECT_SOURCE_DIR}/json-3.11.2.tar.gz
    ${PROJECT_BINARY_DIR}/json-3.11.2.tar.gz
    /tmp/json-3.11.2.tar.gz
    /star-fj/fangjun/download/github/json-3.11.2.tar.gz
  )

  foreach(f IN LISTS possible_file_locations)
    if(EXISTS ${f})
      set(json_URL  "${f}")
      file(TO_CMAKE_PATH "${json_URL}" json_URL)
      message(STATUS "Found local downloaded json: ${json_URL}")
      set(json_URL2)
      break()
    endif()
  endforeach()

  FetchContent_Declare(json
    URL
      ${json_URL}
      ${json_URL2}
    URL_HASH          ${json_HASH}
  )

  FetchContent_GetProperties(json)
  if(NOT json_POPULATED)
    message(STATUS "Downloading json from ${json_URL}")
    FetchContent_Populate(json)
  endif()
  message(STATUS "json is downloaded to ${json_SOURCE_DIR}")
  include_directories(${json_SOURCE_DIR}/include)
  # Use #include "nlohmann/json.hpp"
endfunction()

download_json()
