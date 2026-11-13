include_guard(GLOBAL)

function(download_kaldi_native_io)
  include(FetchContent)

  set(kaldi_native_io_URL  "https://github.com/csukuangfj/kaldi_native_io/archive/refs/tags/v1.22.1.tar.gz")
  set(kaldi_native_io_HASH "SHA256=de8ad3398162870b4e1f3ac78101af209c981a8628555710ed4f1adbfed0af43")

  set(KALDI_NATIVE_IO_BUILD_TESTS OFF CACHE BOOL "" FORCE)
  set(KALDI_NATIVE_IO_BUILD_PYTHON OFF CACHE BOOL "" FORCE)

  # If you don't have access to the Internet,
  # please pre-download kaldi_native_io
  set(possible_file_locations
    $ENV{HOME}/Downloads/kaldi_native_io-1.22.1.tar.gz
    ${CMAKE_SOURCE_DIR}/kaldi_native_io-1.22.1.tar.gz
    ${CMAKE_BINARY_DIR}/kaldi_native_io-1.22.1.tar.gz
    /tmp/kaldi_native_io-1.22.1.tar.gz
    /star-fj/fangjun/download/github/kaldi_native_io-1.22.1.tar.gz
  )

  foreach(f IN LISTS possible_file_locations)
    if(EXISTS ${f})
      set(kaldi_native_io_URL  "${f}")
      file(TO_CMAKE_PATH "${kaldi_native_io_URL}" kaldi_native_io_URL)
      message(STATUS "Found local downloaded kaldi_native_io: ${kaldi_native_io_URL}")
      break()
    endif()
  endforeach()

  FetchContent_Declare(kaldi_native_io
    URL
      ${kaldi_native_io_URL}
    URL_HASH          ${kaldi_native_io_HASH}
  )

  FetchContent_GetProperties(kaldi_native_io)
  if(NOT kaldi_native_io_POPULATED)
    message(STATUS "Downloading kaldi_native_io from ${kaldi_native_io_URL}")
    FetchContent_Populate(kaldi_native_io)
  endif()
  message(STATUS "kaldi_native_io is downloaded to ${kaldi_native_io_SOURCE_DIR}")
  message(STATUS "kaldi_native_io's binary dir is ${kaldi_native_io_BINARY_DIR}")

  include_directories(${kaldi_native_io_SOURCE_DIR})

  if(BUILD_SHARED_LIBS)
    set(_build_shared_libs_bak ${BUILD_SHARED_LIBS})
    set(BUILD_SHARED_LIBS OFF)
  endif()

  add_subdirectory(${kaldi_native_io_SOURCE_DIR} ${kaldi_native_io_BINARY_DIR} EXCLUDE_FROM_ALL)

  if(_build_shared_libs_bak)
    set_target_properties(
        kaldi_native_io_core
      PROPERTIES
        POSITION_INDEPENDENT_CODE ON
        C_VISIBILITY_PRESET hidden
        CXX_VISIBILITY_PRESET hidden
    )
    set(BUILD_SHARED_LIBS ON)
  endif()

  target_include_directories(kaldi_native_io_core
    INTERFACE
      ${kaldi_native_io_SOURCE_DIR}/
  )
  set_target_properties(kaldi_native_io_core PROPERTIES OUTPUT_NAME "kaldi-native-io-core")
  if(NOT BUILD_SHARED_LIBS)
    install(TARGETS
      kaldi_native_io_core
    DESTINATION lib)
  endif()
endfunction()

download_kaldi_native_io()
