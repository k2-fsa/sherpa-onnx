# See also https://github.com/cmake-js/cmake-js
# npm install cmake-js
# ./node_modules/.bin/cmake-js --help
# ./node_modules/.bin/cmake-js --version
# ./node_modules/.bin/cmake-js compile --help
# ./node_modules/.bin/cmake-js compile --log-level
# ./node_modules/.bin/cmake-js compile --log-level verbose
cmake_minimum_required(VERSION 3.15)
cmake_policy(SET CMP0091 NEW)
cmake_policy(SET CMP0042 NEW)

project(sherpa-onnx)

set(CMAKE_CXX_STANDARD 14)

add_definitions(-DNAPI_VERSION=3)

include_directories(${CMAKE_JS_INC})

set(srcs
  src/non-streaming-asr.cc
  src/non-streaming-tts.cc
  src/sherpa-onnx-node-addon-api.cc
  src/spoken-language-identification.cc
  src/streaming-asr.cc
  src/vad.cc
  src/wave-reader.cc
  src/wave-writer.cc
)

if(NOT DEFINED ENV{SHERPA_ONNX_INSTALL_DIR})
  message(FATAL_ERROR "
Please run:
git clone https://github.com/k2-fsa/sherpa-onnx
cd sherpa-onnx
mkdir build
cd build
cmake -DBUILD_SHARED_LIBS=ON -DCMAKE_INSTALL_PREFIX=./install ..
make install
export SHERPA_ONNX_INSTALL_DIR=$PWD/install
  ")
endif()

include_directories($ENV{SHERPA_ONNX_INSTALL_DIR}/include)

# See https://nodejs.github.io/node-addon-examples/build-tools/cmake-js
# Include Node-API wrappers
execute_process(
  COMMAND node -p "require('node-addon-api').include"
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    OUTPUT_VARIABLE NODE_ADDON_API_DIR
)

string(REPLACE "\n" "" NODE_ADDON_API_DIR ${NODE_ADDON_API_DIR})
string(REPLACE "\"" "" NODE_ADDON_API_DIR ${NODE_ADDON_API_DIR})
include_directories(${NODE_ADDON_API_DIR})

link_directories($ENV{SHERPA_ONNX_INSTALL_DIR}/lib)

add_library(${PROJECT_NAME} SHARED ${srcs} ${CMAKE_JS_SRC})
set_target_properties(${PROJECT_NAME} PROPERTIES PREFIX "" SUFFIX ".node")
target_link_libraries(${PROJECT_NAME} ${CMAKE_JS_LIB})

target_link_libraries(${PROJECT_NAME}
  sherpa-onnx-c-api
  sherpa-onnx-core
  kaldi-decoder-core
  sherpa-onnx-kaldifst-core
  sherpa-onnx-fstfar
  sherpa-onnx-fst
  kaldi-native-fbank-core
  piper_phonemize
  espeak-ng
  ucd
  onnxruntime
  -Wl,-rpath,$ENV{SHERPA_ONNX_INSTALL_DIR}/lib
)


if(MSVC AND CMAKE_JS_NODELIB_DEF AND CMAKE_JS_NODELIB_TARGET)
  # Generate node.lib
  execute_process(COMMAND ${CMAKE_AR} /def:${CMAKE_JS_NODELIB_DEF} /out:${CMAKE_JS_NODELIB_TARGET} ${CMAKE_STATIC_LINKER_FLAGS})
endif()
