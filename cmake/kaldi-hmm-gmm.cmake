include_guard(GLOBAL)

function(download_kaldi_hmm_gmm)
  include(FetchContent)

  set(kaldi_hmm_gmm_URL  "https://github.com/csukuangfj/kaldi-hmm-gmm/archive/refs/tags/v1.2.1.tar.gz")
  set(kaldi_hmm_gmm_HASH "SHA256=8344681ef464373bab54d548877ffa1cc82c16097e4f0975ce7b497c2e5b6e33")

  set(KHG_ENABLE_TESTS OFF CACHE BOOL "" FORCE)
  set(KHG_BUILD_PYTHON OFF CACHE BOOL "" FORCE)
  set(KHG_INSTALL_CORE_LIB OFF CACHE BOOL "" FORCE)

  # If you don't have access to the Internet,
  # please pre-download kaldi-hmm-gmm
  set(possible_file_locations
    $ENV{HOME}/Downloads/kaldi-hmm-gmm-1.2.1.tar.gz
    ${CMAKE_SOURCE_DIR}/kaldi-hmm-gmm-1.2.1.tar.gz
    ${CMAKE_BINARY_DIR}/kaldi-hmm-gmm-1.2.1.tar.gz
    /tmp/kaldi-hmm-gmm-1.2.1.tar.gz
    /star-fj/fangjun/download/github/kaldi-hmm-gmm-1.2.1.tar.gz
  )

  foreach(f IN LISTS possible_file_locations)
    if(EXISTS ${f})
      set(kaldi_hmm_gmm_URL  "${f}")
      file(TO_CMAKE_PATH "${kaldi_hmm_gmm_URL}" kaldi_hmm_gmm_URL)
      message(STATUS "Found local downloaded kaldi-hmm-gmm: ${kaldi_hmm_gmm_URL}")
      break()
    endif()
  endforeach()

  FetchContent_Declare(kaldi_hmm_gmm
    URL
      ${kaldi_hmm_gmm_URL}
    URL_HASH          ${kaldi_hmm_gmm_HASH}
  )

  FetchContent_GetProperties(kaldi_hmm_gmm)
  if(NOT kaldi_hmm_gmm_POPULATED)
    message(STATUS "Downloading kaldi-hmm-gmm from ${kaldi_hmm_gmm_URL}")
    FetchContent_Populate(kaldi_hmm_gmm)
  endif()
  message(STATUS "kaldi-hmm-gmm is downloaded to ${kaldi_hmm_gmm_SOURCE_DIR}")
  message(STATUS "kaldi-hmm-gmm's binary dir is ${kaldi_hmm_gmm_BINARY_DIR}")

  include_directories(${kaldi_hmm_gmm_SOURCE_DIR})

  if(BUILD_SHARED_LIBS)
    set(_build_shared_libs_bak ${BUILD_SHARED_LIBS})
    set(BUILD_SHARED_LIBS OFF)
  endif()

  add_subdirectory(${kaldi_hmm_gmm_SOURCE_DIR} ${kaldi_hmm_gmm_BINARY_DIR} EXCLUDE_FROM_ALL)

  if(_build_shared_libs_bak)
    set_target_properties(
        kaldi-hmm-gmm-core
      PROPERTIES
        POSITION_INDEPENDENT_CODE ON
        C_VISIBILITY_PRESET hidden
        CXX_VISIBILITY_PRESET hidden
    )
    set(BUILD_SHARED_LIBS ON)
  endif()

  if(WIN32 AND MSVC)
    target_compile_options(kaldi-hmm-gmm-core PUBLIC
      /wd4018
      /wd4291
    )
  endif()

  # Like kaldi-decoder-core: the static archive is part of the install tree
  # so downstream users can link sherpa-onnx-core with all its transitive
  # dependencies. (KHG_INSTALL_CORE_LIB stays OFF above: the installs of
  # kaldi-hmm-gmm's own subprojects must not run here.)
  if(NOT BUILD_SHARED_LIBS)
    install(TARGETS
      kaldi-hmm-gmm-core
    DESTINATION lib)
  endif()
endfunction()

download_kaldi_hmm_gmm()
