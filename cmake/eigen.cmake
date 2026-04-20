function(download_eigen)
  include(FetchContent)

  set(eigen_URL  "https://gitlab.com/libeigen/eigen/-/archive/5.0.1/eigen-5.0.1.tar.gz")
  set(eigen_HASH "SHA256=e9c326dc8c05cd1e044c71f30f1b2e34a6161a3b6ecf445d56b53ff1669e3dec")

  # If you don't have access to the Internet,
  # please pre-download eigen
  set(possible_file_locations
    $ENV{HOME}/Downloads/eigen-5.0.1.tar.gz
    ${CMAKE_SOURCE_DIR}/eigen-5.0.1.tar.gz
    ${CMAKE_BINARY_DIR}/eigen-5.0.1.tar.gz
    /tmp/eigen-5.0.1.tar.gz
    /star-fj/fangjun/download/github/eigen-5.0.1.tar.gz
  )

  foreach(f IN LISTS possible_file_locations)
    if(EXISTS ${f})
      set(eigen_URL  "${f}")
      file(TO_CMAKE_PATH "${eigen_URL}" eigen_URL)
      message(STATUS "Found local downloaded eigen: ${eigen_URL}")
      break()
    endif()
  endforeach()

  set(BUILD_TESTING OFF CACHE BOOL "" FORCE)
  set(EIGEN_BUILD_DOC OFF CACHE BOOL "" FORCE)

  FetchContent_Declare(eigen
    URL               ${eigen_URL}
    URL_HASH          ${eigen_HASH}
  )

  FetchContent_GetProperties(eigen)
  if(NOT eigen_POPULATED)
    message(STATUS "Downloading eigen from ${eigen_URL}")
    FetchContent_Populate(eigen)
  endif()
  message(STATUS "eigen is downloaded to ${eigen_SOURCE_DIR}")
  message(STATUS "eigen's binary dir is ${eigen_BINARY_DIR}")

  add_subdirectory(${eigen_SOURCE_DIR} ${eigen_BINARY_DIR} EXCLUDE_FROM_ALL)
endfunction()

download_eigen()

