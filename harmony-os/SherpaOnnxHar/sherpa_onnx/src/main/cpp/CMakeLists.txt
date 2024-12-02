# the minimum version of CMake.
cmake_minimum_required(VERSION 3.13.0)
project(myNpmLib)

if (NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17 CACHE STRING "The C++ version to use")
endif()

# Disable warning about
#
# "The DOWNLOAD_EXTRACT_TIMESTAMP option was not given and policy CMP0135 is
#  not set.
if (CMAKE_VERSION VERSION_GREATER_EQUAL "3.24.0")
  cmake_policy(SET CMP0135 NEW)
endif()

set(NATIVERENDER_ROOT_PATH ${CMAKE_CURRENT_SOURCE_DIR})

if(DEFINED PACKAGE_FIND_FILE)
    include(${PACKAGE_FIND_FILE})
endif()

include_directories(${NATIVERENDER_ROOT_PATH}
                    ${NATIVERENDER_ROOT_PATH}/include)

include(FetchContent)
FetchContent_Declare(node_addon_api
    GIT_REPOSITORY "https://github.com/nodejs/node-addon-api.git"
    GIT_TAG c679f6f4c9dc6bf9fc0d99cbe5982bd24a5e2c7b
    PATCH_COMMAND git checkout . && git apply --ignore-whitespace "${CMAKE_CURRENT_LIST_DIR}/my-patch.diff"
)
FetchContent_MakeAvailable(node_addon_api)
FetchContent_GetProperties(node_addon_api)
if(NOT node_addon_api_POPULATED)
    message(STATUS "Downloading node-addon-api from")
    FetchContent_Populate(node_addon_api)
endif()

message(STATUS "node-addon-api is downloaded to ${node_addon_api_SOURCE_DIR}")
include_directories(${node_addon_api_SOURCE_DIR})

add_library(sherpa_onnx SHARED
  audio-tagging.cc
  keyword-spotting.cc
  non-streaming-asr.cc
  non-streaming-speaker-diarization.cc
  non-streaming-tts.cc
  punctuation.cc
  sherpa-onnx-node-addon-api.cc
  speaker-identification.cc
  spoken-language-identification.cc
  streaming-asr.cc
  utils.cc
  vad.cc
  wave-reader.cc
  wave-writer.cc
)

add_library(sherpa_onnx_c_api SHARED IMPORTED)
set_target_properties(sherpa_onnx_c_api
    PROPERTIES
    IMPORTED_LOCATION ${CMAKE_CURRENT_SOURCE_DIR}/libs/${OHOS_ARCH}/libsherpa-onnx-c-api.so)

add_library(onnxruntime SHARED IMPORTED)
set_target_properties(onnxruntime
    PROPERTIES
    IMPORTED_LOCATION ${CMAKE_CURRENT_SOURCE_DIR}/libs/${OHOS_ARCH}/libonnxruntime.so)


target_link_libraries(sherpa_onnx PUBLIC libace_napi.z.so
 libhilog_ndk.z.so # for hilog
 librawfile.z.so
 sherpa_onnx_c_api onnxruntime
)
