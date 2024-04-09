# Copyright (c)  2020  Xiaomi Corporation (author: Fangjun Kuang)

function(download_openfst)
  include(FetchContent)

  set(openfst_URL  "https://github.com/csukuangfj/openfst/archive/f6a37911d53d99d46f74e9cd71c3f5071876ba2d.zip")
  set(openfst_URL2 "https://hub.nuaa.cf/csukuangfj/openfst/archive/f6a37911d53d99d46f74e9cd71c3f5071876ba2d.zip")
  set(openfst_HASH "SHA256=b538ace0a28b21c81f997876797b7e54807cd993eec758c7c690e3e022d93e44")

  # If you don't have access to the Internet,
  # please pre-download it
  set(possible_file_locations
    $ENV{HOME}/Downloads/openfst-f6a37911d53d99d46f74e9cd71c3f5071876ba2d.zip
    ${CMAKE_SOURCE_DIR}/openfst-f6a37911d53d99d46f74e9cd71c3f5071876ba2d.zip
    ${CMAKE_BINARY_DIR}/openfst-f6a37911d53d99d46f74e9cd71c3f5071876ba2d.zip
    /tmp/openfst-f6a37911d53d99d46f74e9cd71c3f5071876ba2d.zip
    /star-fj/fangjun/download/github/openfst-f6a37911d53d99d46f74e9cd71c3f5071876ba2d.zip
  )

  foreach(f IN LISTS possible_file_locations)
    if(EXISTS ${f})
      set(openfst_URL  "${f}")
      file(TO_CMAKE_PATH "${openfst_URL}" openfst_URL)
      set(openfst_URL2)
      break()
    endif()
  endforeach()

  set(HAVE_BIN OFF CACHE BOOL "" FORCE)
  set(HAVE_SCRIPT ON CACHE BOOL "" FORCE)
  set(HAVE_COMPACT OFF CACHE BOOL "" FORCE)
  set(HAVE_COMPRESS OFF CACHE BOOL "" FORCE)
  set(HAVE_CONST OFF CACHE BOOL "" FORCE)
  set(HAVE_FAR ON CACHE BOOL "" FORCE)
  set(HAVE_GRM OFF CACHE BOOL "" FORCE)
  set(HAVE_PDT OFF CACHE BOOL "" FORCE)
  set(HAVE_MPDT OFF CACHE BOOL "" FORCE)
  set(HAVE_LINEAR OFF CACHE BOOL "" FORCE)
  set(HAVE_LOOKAHEAD OFF CACHE BOOL "" FORCE)
  set(HAVE_NGRAM OFF CACHE BOOL "" FORCE)
  set(HAVE_PYTHON OFF CACHE BOOL "" FORCE)
  set(HAVE_SPECIAL OFF CACHE BOOL "" FORCE)

  if(NOT WIN32)
    FetchContent_Declare(openfst
      URL
        ${openfst_URL}
        ${openfst_URL2}
      URL_HASH          ${openfst_HASH}
      PATCH_COMMAND
        sed -i.bak s/enable_testing\(\)//g "src/CMakeLists.txt" &&
        sed -i.bak s/add_subdirectory\(test\)//g "src/CMakeLists.txt" &&
        sed -i.bak /message/d "src/script/CMakeLists.txt"
        # sed -i.bak s/add_subdirectory\(script\)//g "src/CMakeLists.txt" &&
        # sed -i.bak s/add_subdirectory\(extensions\)//g "src/CMakeLists.txt"
    )
  else()
    FetchContent_Declare(openfst
      URL               ${openfst_URL}
      URL_HASH          ${openfst_HASH}
    )
  endif()

  FetchContent_GetProperties(openfst)
  if(NOT openfst_POPULATED)
    message(STATUS "Downloading openfst from ${openfst_URL}")
    FetchContent_Populate(openfst)
  endif()
  message(STATUS "openfst is downloaded to ${openfst_SOURCE_DIR}")
  add_subdirectory(${openfst_SOURCE_DIR} ${openfst_BINARY_DIR} EXCLUDE_FROM_ALL)
  set(openfst_SOURCE_DIR ${openfst_SOURCE_DIR} PARENT_SCOPE)

  # Rename libfst.so.6 to libsherpa-onnx-fst.so.6 to avoid potential conflicts
  # when sherpa-onnx is installed.
  set_target_properties(fst PROPERTIES OUTPUT_NAME "sherpa-onnx-fst")
  set_target_properties(fstfar PROPERTIES OUTPUT_NAME "sherpa-onnx-fstfar")

  if(LINUX)
    target_compile_options(fst PUBLIC -Wno-missing-template-keyword)
  endif()

  target_include_directories(fst
    PUBLIC
      ${openfst_SOURCE_DIR}/src/include
  )

  target_include_directories(fstfar
    PUBLIC
      ${openfst_SOURCE_DIR}/src/include
  )
  # installed in ./kaldi-decoder.cmake
endfunction()

download_openfst()
