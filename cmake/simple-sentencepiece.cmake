function(download_simple_sentencepiece)
  include(FetchContent)

  set(simple-sentencepiece_URL "https://github.com/pkufool/simple-sentencepiece/archive/refs/tags/v0.7.tar.gz")
  set(simple-sentencepiece_URL2 "https://hub.nauu.cf/pkufool/simple-sentencepiece/archive/refs/tags/v0.7.tar.gz")
  set(simple-sentencepiece_HASH "SHA256=1748a822060a35baa9f6609f84efc8eb54dc0e74b9ece3d82367b7119fdc75af")

  # If you don't have access to the Internet,
  # please pre-download simple-sentencepiece
  set(possible_file_locations
    $ENV{HOME}/Downloads/simple-sentencepiece-0.7.tar.gz
    ${CMAKE_SOURCE_DIR}/simple-sentencepiece-0.7.tar.gz
    ${CMAKE_BINARY_DIR}/simple-sentencepiece-0.7.tar.gz
    /tmp/simple-sentencepiece-0.7.tar.gz
    /star-fj/fangjun/download/github/simple-sentencepiece-0.7.tar.gz
  )

  foreach(f IN LISTS possible_file_locations)
    if(EXISTS ${f})
      set(simple-sentencepiece_URL  "${f}")
      file(TO_CMAKE_PATH "${simple-sentencepiece_URL}" simple-sentencepiece_URL)
      message(STATUS "Found local downloaded simple-sentencepiece: ${simple-sentencepiece_URL}")
      set(simple-sentencepiece_URL2)
      break()
    endif()
  endforeach()

  set(SBPE_ENABLE_TESTS OFF CACHE BOOL "" FORCE)
  set(SBPE_BUILD_PYTHON OFF CACHE BOOL "" FORCE)

  FetchContent_Declare(simple-sentencepiece
    URL
      ${simple-sentencepiece_URL}
      ${simple-sentencepiece_URL2}
    URL_HASH
      ${simple-sentencepiece_HASH}
  )

  FetchContent_GetProperties(simple-sentencepiece)
  if(NOT simple-sentencepiece_POPULATED)
    message(STATUS "Downloading simple-sentencepiece ${simple-sentencepiece_URL}")
    FetchContent_Populate(simple-sentencepiece)
  endif()
  message(STATUS "simple-sentencepiece is downloaded to ${simple-sentencepiece_SOURCE_DIR}")
  add_subdirectory(${simple-sentencepiece_SOURCE_DIR} ${simple-sentencepiece_BINARY_DIR} EXCLUDE_FROM_ALL)

  target_include_directories(ssentencepiece_core
    PUBLIC
      ${simple-sentencepiece_SOURCE_DIR}/
  )

  if(SHERPA_ONNX_ENABLE_PYTHON AND WIN32)
    install(TARGETS ssentencepiece_core DESTINATION ..)
  else()
    install(TARGETS ssentencepiece_core DESTINATION lib)
  endif()

  if(WIN32 AND BUILD_SHARED_LIBS)
    install(TARGETS ssentencepiece_core DESTINATION bin)
  endif()
endfunction()

download_simple_sentencepiece()
