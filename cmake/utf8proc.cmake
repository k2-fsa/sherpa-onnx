function(download_utf8proc)
  include(FetchContent)

  # utf8proc owns all Unicode handling here: NFC/NFD normalization, simple
  # case mapping and general categories. Keeping that in a library that tracks
  # the Unicode standard beats maintaining our own case/composition tables.
  set(utf8proc_URL   "https://github.com/JuliaStrings/utf8proc/archive/refs/tags/v2.12.0.tar.gz")
  set(utf8proc_HASH  "SHA256=a393fbef160835fb315bc3e91ba8d86f7a73a7cec9e6198b6c60b848b498bfeb")

  set(UTF8PROC_INSTALL OFF CACHE BOOL "" FORCE)
  set(UTF8PROC_ENABLE_TESTING OFF CACHE BOOL "" FORCE)

  # If you don't have access to the Internet,
  # please pre-download utf8proc
  set(possible_file_locations
    $ENV{HOME}/Downloads/utf8proc-2.12.0.tar.gz
    ${CMAKE_SOURCE_DIR}/utf8proc-2.12.0.tar.gz
    ${CMAKE_BINARY_DIR}/utf8proc-2.12.0.tar.gz
    /tmp/utf8proc-2.12.0.tar.gz
    /star-fj/fangjun/download/github/utf8proc-2.12.0.tar.gz
  )

  foreach(f IN LISTS possible_file_locations)
    if(EXISTS ${f})
      set(utf8proc_URL "${f}")
      file(TO_CMAKE_PATH "${utf8proc_URL}" utf8proc_URL)
      message(STATUS "Found local downloaded utf8proc: ${utf8proc_URL}")
      break()
    endif()
  endforeach()

  FetchContent_Declare(utf8proc
    URL
      ${utf8proc_URL}
    URL_HASH          ${utf8proc_HASH}
  )

  FetchContent_GetProperties(utf8proc)
  if(NOT utf8proc_POPULATED)
    message(STATUS "Downloading utf8proc from ${utf8proc_URL}")
    FetchContent_Populate(utf8proc)
  endif()
  message(STATUS "utf8proc is downloaded to ${utf8proc_SOURCE_DIR}")

  if(BUILD_SHARED_LIBS)
    set(_build_shared_libs_bak ${BUILD_SHARED_LIBS})
    set(BUILD_SHARED_LIBS OFF)
  endif()

  add_subdirectory(${utf8proc_SOURCE_DIR} ${utf8proc_BINARY_DIR} EXCLUDE_FROM_ALL)

  if(_build_shared_libs_bak)
    set_target_properties(utf8proc
      PROPERTIES
        POSITION_INDEPENDENT_CODE ON
        C_VISIBILITY_PRESET hidden
    )
    set(BUILD_SHARED_LIBS ON)
  endif()

  if(NOT BUILD_SHARED_LIBS)
    install(TARGETS utf8proc DESTINATION lib)
  endif()
endfunction()

download_utf8proc()
