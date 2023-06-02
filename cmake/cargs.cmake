function(download_cargs)
  include(FetchContent)

  set(cargs_URL "https://github.com/likle/cargs/archive/refs/tags/v1.0.3.tar.gz")

  FetchContent_Declare(cargs URL ${cargs_URL} )

  FetchContent_GetProperties(cargs)
  if(NOT cargs_POPULATED)
    message(STATUS "Downloading cargs ${cargs_URL}")
    FetchContent_Populate(cargs)
  endif()
  message(STATUS "cargs is downloaded to ${cargs_SOURCE_DIR}")
  add_subdirectory(${cargs_SOURCE_DIR} ${cargs_BINARY_DIR} EXCLUDE_FROM_ALL)
endfunction()

download_cargs()
