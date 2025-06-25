message(STATUS "CMAKE_SOURCE_DIR: ${CMAKE_SOURCE_DIR}")
message(STATUS "CMAKE_BINARY_DIR: ${CMAKE_BINARY_DIR}")
message(STATUS "PROJECT_SOURCE_DIR: ${PROJECT_SOURCE_DIR}")
message(STATUS "PROJECT_BINARY_DIR: ${PROJECT_BINARY_DIR}")
message(STATUS "CMake version: ${CMAKE_VERSION}")
message(STATUS "CMAKE_SYSTEM: ${CMAKE_SYSTEM}")
message(STATUS "CMAKE_SYSTEM_NAME: ${CMAKE_SYSTEM_NAME}")
message(STATUS "CMAKE_SYSTEM_VERSION: ${CMAKE_SYSTEM_VERSION}")
message(STATUS "CMAKE_SYSTEM_PROCESSOR: ${CMAKE_SYSTEM_PROCESSOR}")

find_package(Git QUIET)
if(Git_FOUND)
  execute_process(COMMAND
    "${GIT_EXECUTABLE}" describe --always --abbrev=40
    WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
    OUTPUT_VARIABLE SHERPA_ONNX_GIT_SHA1
    ERROR_QUIET OUTPUT_STRIP_TRAILING_WHITESPACE
  )

  execute_process(COMMAND
    "${GIT_EXECUTABLE}" log -1 --format=%ad --date=local
    WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
    OUTPUT_VARIABLE SHERPA_ONNX_GIT_DATE
    ERROR_QUIET OUTPUT_STRIP_TRAILING_WHITESPACE
  )
  message(STATUS "sherpa-onnx git sha1: ${SHERPA_ONNX_GIT_SHA1}")
  message(STATUS "sherpa-onnx git date: ${SHERPA_ONNX_GIT_DATE}")
else()
  message(WARNING "git is not found")
endif()

if(UNIX AND NOT APPLE)
  execute_process(COMMAND
    lsb_release -sd
    OUTPUT_VARIABLE SHERPA_ONNX_OS
    OUTPUT_STRIP_TRAILING_WHITESPACE
  )
elseif(APPLE)
  execute_process(COMMAND
    sw_vers -productName
    OUTPUT_VARIABLE _product_name
    OUTPUT_STRIP_TRAILING_WHITESPACE
  )

  execute_process(COMMAND
    sw_vers -productVersion
    OUTPUT_VARIABLE _product_version
    OUTPUT_STRIP_TRAILING_WHITESPACE
  )

  execute_process(COMMAND
    sw_vers -buildVersion
    OUTPUT_VARIABLE _build_version
    OUTPUT_STRIP_TRAILING_WHITESPACE
  )
  set(SHERPA_ONNX_OS "${_product_name} ${_product_version} ${_build_version}")
elseif(WIN32)
  execute_process(COMMAND
    wmic os get caption,version
    OUTPUT_VARIABLE SHERPA_ONNX_OS_TWO_LINES
    OUTPUT_STRIP_TRAILING_WHITESPACE
  )
  # Now SHERPA_ONNX_OS_TWO_LINES contains something like
  #  Caption                          Version
  #  Microsoft Windows 10 Pro         10.0.18362
  string(REPLACE "\n" ";" SHERPA_ONNX_OS_LIST ${SHERPA_ONNX_OS_TWO_LINES})
  list(GET SHERPA_ONNX_OS_LIST 1 SHERPA_ONNX_OS)
else()
  set(SHERPA_ONNX_OS "Unknown")
endif()
message(STATUS "OS used to build sherpa-onnx: ${SHERPA_ONNX_OS}")

if(CMAKE_CXX_COMPILER)
  message(STATUS "C++ compiler: ${CMAKE_CXX_COMPILER}")
  if(CMAKE_CXX_COMPILER_ID)
    message(STATUS "C++ compiler ID: ${CMAKE_CXX_COMPILER_ID}")
    message(STATUS "C++ compiler version: ${CMAKE_CXX_COMPILER_VERSION}")
  endif()
endif()

if(CMAKE_C_COMPILER)
  message(STATUS "C compiler: ${CMAKE_C_COMPILER}")
  if(CMAKE_C_COMPILER_ID)
    message(STATUS "C compiler ID: ${CMAKE_C_COMPILER_ID}")
    message(STATUS "C compiler version: ${CMAKE_C_COMPILER_VERSION}")
  endif()
endif()
