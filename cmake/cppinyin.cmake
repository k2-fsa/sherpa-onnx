function(download_cppinyin)
  include(FetchContent)

  set(cppinyin_URL "https://github.com/pkufool/cppinyin/archive/refs/tags/v0.10.tar.gz")
  set(cppinyin_URL2 "https://gh-proxy.com/https://github.com/pkufool/cppinyin/archive/refs/tags/v0.10.tar.gz")
  set(cppinyin_HASH "SHA256=abe6584d7ee56829e8f4b5fbda3b50ecdf49a13be8e413a78d1b0d5d5c019982")

  # If you don't have access to the Internet,
  # please pre-download cppinyin
  set(possible_file_locations
    $ENV{HOME}/Downloads/cppinyin-0.10.tar.gz
    ${CMAKE_SOURCE_DIR}/cppinyin-0.10.tar.gz
    ${CMAKE_BINARY_DIR}/cppinyin-0.10.tar.gz
    /tmp/cppinyin-0.10.tar.gz
    /star-fj/fangjun/download/github/cppinyin-0.10.tar.gz
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

    if(WIN32)
      set(PATCH_FILE "${CMAKE_SOURCE_DIR}/cmake/cppinyin.patch")
      message(STATUS "cppinyin patch file ${PATCH_FILE}")

      if(NOT EXISTS ${PATCH_FILE})
        message(FATAL_ERROR "${PATCH_FILE} does not exist")
      endif()

      find_program(GIT_EXECUTABLE git)
      if(NOT GIT_EXECUTABLE)
        message(FATAL_ERROR "Cannot find the command: git")
      endif()

      execute_process(
          COMMAND ${GIT_EXECUTABLE} apply --verbose "${PATCH_FILE}"
          WORKING_DIRECTORY ${cppinyin_SOURCE_DIR}
          RESULT_VARIABLE PATCH_RESULT
          ERROR_VARIABLE PATCH_ERROR
          OUTPUT_VARIABLE PATCH_OUTPUT
      )
      if(PATCH_RESULT EQUAL 0)
        message(STATUS "Applied ${PATCH_FILE}")
        message(STATUS "Output from git apply：${PATCH_OUTPUT}")
      else()
          message(FATAL_ERROR "Patch failed：${PATCH_ERROR}")
      endif()
    endif()
  endif()

  message(STATUS "cppinyin is downloaded to ${cppinyin_SOURCE_DIR}")



  if(BUILD_SHARED_LIBS)
    set(_build_shared_libs_bak ${BUILD_SHARED_LIBS})
    set(BUILD_SHARED_LIBS OFF)
  endif()

  add_subdirectory(${cppinyin_SOURCE_DIR} ${cppinyin_BINARY_DIR} EXCLUDE_FROM_ALL)

  if(_build_shared_libs_bak)
    set_target_properties(cppinyin_core
      PROPERTIES
        POSITION_INDEPENDENT_CODE ON
        C_VISIBILITY_PRESET hidden
        CXX_VISIBILITY_PRESET hidden
    )
    set(BUILD_SHARED_LIBS ON)
  endif()

  target_include_directories(cppinyin_core
    PUBLIC
      ${cppinyin_SOURCE_DIR}/
  )

  if(NOT BUILD_SHARED_LIBS)
    install(TARGETS cppinyin_core DESTINATION lib)
  endif()

endfunction()

download_cppinyin()
