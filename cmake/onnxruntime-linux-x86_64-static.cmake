# Copyright (c)  2022-2023  Xiaomi Corporation
message(STATUS "CMAKE_SYSTEM_NAME: ${CMAKE_SYSTEM_NAME}")
message(STATUS "CMAKE_SYSTEM_PROCESSOR: ${CMAKE_SYSTEM_PROCESSOR}")

if(NOT CMAKE_SYSTEM_NAME STREQUAL Linux)
  message(FATAL_ERROR "This file is for Linux only. Given: ${CMAKE_SYSTEM_NAME}")
endif()

if(NOT CMAKE_SYSTEM_PROCESSOR STREQUAL x86_64)
  message(FATAL_ERROR "This file is for x86_64 only. Given: ${CMAKE_SYSTEM_PROCESSOR}")
endif()

if(BUILD_SHARED_LIBS)
  message(FATAL_ERROR "This file is for building static libraries. BUILD_SHARED_LIBS: ${BUILD_SHARED_LIBS}")
endif()

# Note: we use the glibc2_28 package, not glibc2_17.
#
# The glibc2_17 x86_64 static package is built in onnxruntime-build's
# linux-217.yaml with Red Hat SCL devtoolset-11, whose gcc is configured
# --with-default-libstdcxx-abi=gcc4-compatible, so it produces
# _GLIBCXX_USE_CXX11_ABI=0 (old COW std::string). Every other package we
# ship -- the aarch64 glibc2_17 and the x86_64 glibc2_28 builds -- is =1.
#
# Mixing the two corrupts the heap at startup with "free(): invalid
# pointer": std::filesystem::path's non-inline members (_M_split_cmpts,
# operator=) live in libstdc++ and are laid out for =1, while the path
# objects in the archive are laid out for =0. Linking libstdc++ statically
# does not help; the layout disagreement is independent of that.
#
# See also https://github.com/k2-fsa/sherpa-onnx/issues/3513
#
# TODO: once onnxruntime-build ships a glibc2_17 build made with
# _GLIBCXX_USE_CXX11_ABI=1, switch back to it to keep the glibc 2.17 floor
# for x86_64 Linux static binaries.
set(onnxruntime_URL  "https://github.com/csukuangfj/onnxruntime-libs/releases/download/v1.28.2/onnxruntime-linux-x64-static_lib-1.28.2-glibc2_28.zip")
set(onnxruntime_HASH "SHA256=cdff51ef93072792fa20f386e7576fd23b0140be202ad62f5d921e385ffb6103")

# If you don't have access to the Internet,
# please download onnxruntime to one of the following locations.
# You can add more if you want.
set(possible_file_locations
  $ENV{HOME}/Downloads/onnxruntime-linux-x64-static_lib-1.28.2-glibc2_28.zip
  ${CMAKE_SOURCE_DIR}/onnxruntime-linux-x64-static_lib-1.28.2-glibc2_28.zip
  ${CMAKE_BINARY_DIR}/onnxruntime-linux-x64-static_lib-1.28.2-glibc2_28.zip
  /tmp/onnxruntime-linux-x64-static_lib-1.28.2-glibc2_28.zip
  /star-fj/fangjun/download/github/onnxruntime-linux-x64-static_lib-1.28.2-glibc2_28.zip
)

foreach(f IN LISTS possible_file_locations)
  if(EXISTS ${f})
    set(onnxruntime_URL  "${f}")
    file(TO_CMAKE_PATH "${onnxruntime_URL}" onnxruntime_URL)
    message(STATUS "Found local downloaded onnxruntime: ${onnxruntime_URL}")
    break()
  endif()
endforeach()

FetchContent_Declare(onnxruntime
  URL
    ${onnxruntime_URL}
  URL_HASH          ${onnxruntime_HASH}
)

FetchContent_GetProperties(onnxruntime)
if(NOT onnxruntime_POPULATED)
  message(STATUS "Downloading onnxruntime from ${onnxruntime_URL}")
  FetchContent_Populate(onnxruntime)
endif()
message(STATUS "onnxruntime is downloaded to ${onnxruntime_SOURCE_DIR}")

# for static libraries, we use onnxruntime_lib_files directly below
include_directories(${onnxruntime_SOURCE_DIR}/include)

file(GLOB onnxruntime_lib_files "${onnxruntime_SOURCE_DIR}/lib/lib*.a")

set(onnxruntime_lib_files ${onnxruntime_lib_files} PARENT_SCOPE)

message(STATUS "onnxruntime lib files: ${onnxruntime_lib_files}")
install(FILES ${onnxruntime_lib_files} DESTINATION lib)
