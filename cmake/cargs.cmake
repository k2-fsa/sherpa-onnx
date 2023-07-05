function(download_cargs)
  include(FetchContent)

  set(cargs_URL "https://github.com/likle/cargs/archive/refs/tags/v1.0.3.tar.gz")
  set(cargs_HASH "SHA256=ddba25bd35e9c6c75bc706c126001b8ce8e084d40ef37050e6aa6963e836eb8b")

  # If you don't have access to the Internet,
  # please pre-download asio
  set(possible_file_locations
    $ENV{HOME}/Downloads/cargs-v1-0-3.tar.gz
    ${PROJECT_SOURCE_DIR}/cargs-v1-0-3.tar.gz
    ${PROJECT_BINARY_DIR}/cargs-v1-0-3.tar.gz
    /tmp/cargs-v1-0-3.tar.gz
    /star-fj/fangjun/download/github/cargs-v1-0-3.tar.gz
  )

  foreach(f IN LISTS possible_file_locations)
    if(EXISTS ${f})
      set(cargs_URL  "${f}")
      file(TO_CMAKE_PATH "${cargs_URL}" cargs_URL)
      message(STATUS "Found local downloaded cargs: ${cargs_URL}")
      break()
    endif()
  endforeach()

  FetchContent_Declare(cargs URL ${cargs_URL} URL_HASH ${cargs_HASH})

  FetchContent_GetProperties(cargs)
  if(NOT cargs_POPULATED)
    message(STATUS "Downloading cargs ${cargs_URL}")
    FetchContent_Populate(cargs)
  endif()
  message(STATUS "cargs is downloaded to ${cargs_SOURCE_DIR}")
  add_subdirectory(${cargs_SOURCE_DIR} ${cargs_BINARY_DIR} EXCLUDE_FROM_ALL)
endfunction()

download_cargs()
