message(STATUS "CMAKE_SYSTEM_NAME: ${CMAKE_SYSTEM_NAME}")
message(STATUS "CMAKE_SYSTEM_PROCESSOR: ${CMAKE_SYSTEM_PROCESSOR}")

if(NOT CMAKE_SYSTEM_NAME STREQUAL Linux)
  message(FATAL_ERROR "This file is for Linux only. Given: ${CMAKE_SYSTEM_NAME}")
endif()

if(NOT CMAKE_SYSTEM_PROCESSOR STREQUAL riscv64)
  message(FATAL_ERROR "This file is for riscv64 only. Given: ${CMAKE_SYSTEM_PROCESSOR}")
endif()

if(NOT BUILD_SHARED_LIBS)
  message(FATAL_ERROR "This file is for building shared libraries. BUILD_SHARED_LIBS: ${BUILD_SHARED_LIBS}")
endif()

set(onnxruntime_pkg_name "spacemit-ort.riscv64.2.0.1.tar.gz")
set(onnxruntime_URL  "https://archive.spacemit.com/spacemit-ai/onnxruntime/${onnxruntime_pkg_name}")
set(onnxruntime_HASH "SHA256=8a15035aca34d5fd95f24444d4c7843265c1a81f49d84ec6fe9c6d0fdf5b55cf")

# If you don't have access to the Internet,
# please download onnxruntime to one of the following locations.
# You can add more if you want.
set(possible_file_locations
  $ENV{HOME}/Downloads/${onnxruntime_pkg_name}
  ${CMAKE_SOURCE_DIR}/${onnxruntime_pkg_name}
  ${CMAKE_BINARY_DIR}/${onnxruntime_pkg_name}
  /tmp/${onnxruntime_pkg_name}
  /star-fj/fangjun/download/github/${onnxruntime_pkg_name}
)

foreach(f IN LISTS possible_file_locations)
  if(EXISTS ${f})
    set(onnxruntime_URL  "${f}")
    file(TO_CMAKE_PATH "${onnxruntime_URL}" onnxruntime_URL)
    message(STATUS "Found local downloaded onnxruntime: ${onnxruntime_URL}")
    set(onnxruntime_URL2)
    break()
  endif()
endforeach()

FetchContent_Declare(onnxruntime
  URL
    ${onnxruntime_URL}
    ${onnxruntime_URL2}
  URL_HASH          ${onnxruntime_HASH}
)

FetchContent_GetProperties(onnxruntime)
if(NOT onnxruntime_POPULATED)
  message(STATUS "Downloading onnxruntime from ${onnxruntime_URL}")
  FetchContent_Populate(onnxruntime)
endif()
message(STATUS "onnxruntime is downloaded to ${onnxruntime_SOURCE_DIR}")

file(GLOB onnxruntime_lib_files
  "${onnxruntime_SOURCE_DIR}/lib/libonnxruntime*"
  "${onnxruntime_SOURCE_DIR}/lib/libspacemit_ep*"
)
set(onnxruntime_lib_files ${onnxruntime_lib_files} PARENT_SCOPE)
message(STATUS "onnxruntime lib files: ${onnxruntime_lib_files}")
install(FILES ${onnxruntime_lib_files} DESTINATION lib)
