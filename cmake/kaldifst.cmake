include_guard(GLOBAL)

function(download_kaldifst)
  # Already provided by a parent project.
  if(TARGET kaldifst_core)
    return()
  endif()

  include(FetchContent)

  set(kaldifst_URL  "https://github.com/k2-fsa/kaldifst/archive/refs/tags/v1.9.2.tar.gz")
  set(kaldifst_HASH "SHA256=8007cfde794f6c5e287d0e5e22753c1cf3503c9b02512fbcd6dc5d75e5b41fa2")

  # If you don't have access to the Internet,
  # please pre-download kaldifst
  set(possible_file_locations
    $ENV{HOME}/Downloads/kaldifst-1.9.2.tar.gz
    ${CMAKE_SOURCE_DIR}/kaldifst-1.9.2.tar.gz
    ${CMAKE_BINARY_DIR}/kaldifst-1.9.2.tar.gz
    /tmp/kaldifst-1.9.2.tar.gz
    /star-fj/fangjun/download/github/kaldifst-1.9.2.tar.gz
  )

  foreach(f IN LISTS possible_file_locations)
    if(EXISTS ${f})
      set(kaldifst_URL  "${f}")
      file(TO_CMAKE_PATH "${kaldifst_URL}" kaldifst_URL)
      message(STATUS "Found local downloaded kaldifst: ${kaldifst_URL}")
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

  # kaldfist's CMakeLists needs its own cmake/ dir on CMAKE_MODULE_PATH at
  # configure time (include(openfst)); this project's cmake/openfst.cmake
  # shadows it (module path order) and renames the archives.
  list(APPEND CMAKE_MODULE_PATH ${kaldifst_SOURCE_DIR}/cmake)

  if(BUILD_SHARED_LIBS)
    set(_build_shared_libs_bak ${BUILD_SHARED_LIBS})
    set(BUILD_SHARED_LIBS OFF)
  endif()

  add_subdirectory(${kaldifst_SOURCE_DIR} ${kaldifst_BINARY_DIR} EXCLUDE_FROM_ALL)

  if(_build_shared_libs_bak)
    set_target_properties(kaldifst_core
      PROPERTIES
        POSITION_INDEPENDENT_CODE ON
        C_VISIBILITY_PRESET hidden
        CXX_VISIBILITY_PRESET hidden
    )
    set(BUILD_SHARED_LIBS ON)
  endif()

  target_include_directories(kaldifst_core
    PUBLIC
      ${kaldifst_SOURCE_DIR}/
  )

  set_target_properties(kaldifst_core PROPERTIES OUTPUT_NAME "sherpa-onnx-kaldifst-core")

  if(NOT BUILD_SHARED_LIBS)
    install(TARGETS
      kaldifst_core
    DESTINATION lib)
  endif()
endfunction()

download_kaldifst()
