function(download_pybind11)
  include(FetchContent)

  set(pybind11_URL  "https://github.com/pybind/pybind11/archive/refs/tags/v2.10.2.tar.gz")
  set(pybind11_URL2 "https://hub.nuaa.cf/pybind/pybind11/archive/refs/tags/v2.10.2.tar.gz")
  set(pybind11_HASH "SHA256=93bd1e625e43e03028a3ea7389bba5d3f9f2596abc074b068e70f4ef9b1314ae")

  # If you don't have access to the Internet,
  # please pre-download pybind11
  set(possible_file_locations
    $ENV{HOME}/Downloads/pybind11-2.10.2.tar.gz
    ${CMAKE_SOURCE_DIR}/pybind11-2.10.2.tar.gz
    ${CMAKE_BINARY_DIR}/pybind11-2.10.2.tar.gz
    /tmp/pybind11-2.10.2.tar.gz
    /star-fj/fangjun/download/github/pybind11-2.10.2.tar.gz
  )

  foreach(f IN LISTS possible_file_locations)
    if(EXISTS ${f})
      set(pybind11_URL  "${f}")
      file(TO_CMAKE_PATH "${pybind11_URL}" pybind11_URL)
      message(STATUS "Found local downloaded pybind11: ${pybind11_URL}")
      set(pybind11_URL2)
      break()
    endif()
  endforeach()

  FetchContent_Declare(pybind11
    URL
      ${pybind11_URL}
      ${pybind11_URL2}
    URL_HASH          ${pybind11_HASH}
  )

  FetchContent_GetProperties(pybind11)
  if(NOT pybind11_POPULATED)
    message(STATUS "Downloading pybind11 from ${pybind11_URL}")
    FetchContent_Populate(pybind11)
  endif()
  message(STATUS "pybind11 is downloaded to ${pybind11_SOURCE_DIR}")
  add_subdirectory(${pybind11_SOURCE_DIR} ${pybind11_BINARY_DIR} EXCLUDE_FROM_ALL)
endfunction()

download_pybind11()
