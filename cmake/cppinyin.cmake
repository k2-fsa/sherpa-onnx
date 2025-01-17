function(download_cppinyin)
  include(FetchContent)

  set(cppinyin_URL "https://github.com/pkufool/cppinyin/archive/refs/tags/v0.1.tar.gz")
  set(cppinyin_URL2 "https://hub.nuaa.cf/pkufool/cppinyin/archive/refs/tags/v0.1.tar.gz")
  set(cppinyin_HASH "SHA256=3659bc0c28d17d41ce932807c1cdc1da8c861e6acee969b5844d0d0a3c5ef78b")

  # If you don't have access to the Internet,
  # please pre-download cppinyin
  set(possible_file_locations
    $ENV{HOME}/Downloads/cppinyin-0.1.tar.gz
    ${CMAKE_SOURCE_DIR}/cppinyin-0.1.tar.gz
    ${CMAKE_BINARY_DIR}/cppinyin-0.1.tar.gz
    /tmp/cppinyin-0.1.tar.gz
    /star-fj/fangjun/download/github/cppinyin-0.1.tar.gz
  )

  foreach(f IN LISTS possible_file_locations)
    if(EXISTS ${f})
      set(cppinyin_URL  "${f}")
      file(TO_CMAKE_PATH "${cppinyin_URL}" cppinyin_URL)
      message(STATUS "Found local downloaded cppinyin: ${cppinyin_URL}")
      set(cppinyin_URL2)
      break()
    endif()
  endforeach()

  set(CPPINYIN_ENABLE_TESTS OFF CACHE BOOL "" FORCE)
  set(CPPINYIN_BUILD_PYTHON OFF CACHE BOOL "" FORCE)

  FetchContent_Declare(cppinyin
    URL
      ${cppinyin_URL}
      ${cppinyin_URL2}
    URL_HASH
      ${cppinyin_HASH}
  )

  FetchContent_GetProperties(cppinyin)
  if(NOT cppinyin_POPULATED)
    message(STATUS "Downloading cppinyin ${cppinyin_URL}")
    FetchContent_Populate(cppinyin)
  endif()
  message(STATUS "cppinyin is downloaded to ${cppinyin_SOURCE_DIR}")
  add_subdirectory(${cppinyin_SOURCE_DIR} ${cppinyin_BINARY_DIR} EXCLUDE_FROM_ALL)

  target_include_directories(cppinyin_core
    PUBLIC
      ${cppinyin_SOURCE_DIR}/
  )

  if(SHERPA_ONNX_ENABLE_PYTHON AND WIN32)
    install(TARGETS cppinyin_core DESTINATION ..)
  else()
    install(TARGETS cppinyin_core DESTINATION lib)
  endif()

  if(WIN32 AND BUILD_SHARED_LIBS)
    install(TARGETS cppinyin_core DESTINATION bin)
  endif()
endfunction()

download_cppinyin()
