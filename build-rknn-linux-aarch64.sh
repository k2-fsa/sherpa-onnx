#!/usr/bin/env bash

set -ex

# Before you run this file, make sure you have first cloned
# https://github.com/airockchip/rknn-toolkit2
# and set the environment variable SHERPA_ONNX_RKNN_TOOLKIT2_PATH

if [ -z $SHERPA_ONNX_RKNN_TOOLKIT2_PATH ]; then
  SHERPA_ONNX_RKNN_TOOLKIT2_PATH=/star-fj/fangjun/open-source/rknn-toolkit2
fi

if [ ! -d $SHERPA_ONNX_RKNN_TOOLKIT2_PATH ]; then
  echo "Please first clone https://github.com/airockchip/rknn-toolkit2"
  echo "You can use"
  echo " git clone --depth 1 https://github.com/airockchip/rknn-toolkit2"
  echo " export SHERPA_ONNX_RKNN_TOOLKIT2_PATH=$PWD/rknn-toolkit2"

  exit 1
fi

if [ ! -f  $SHERPA_ONNX_RKNN_TOOLKIT2_PATH/rknpu2/runtime/Linux/librknn_api/include/rknn_api.h ]; then
  echo "$SHERPA_ONNX_RKNN_TOOLKIT2_PATH/rknpu2/runtime/Linux/librknn_api/include/rknn_api.h does not exist"
  exit 1
fi

if [ ! -f $SHERPA_ONNX_RKNN_TOOLKIT2_PATH/rknpu2/runtime/Linux/librknn_api/aarch64/librknnrt.so ]; then
  echo "$SHERPA_ONNX_RKNN_TOOLKIT2_PATH/rknpu2/runtime/Linux/librknn_api/aarch64/librknnrt.so does not exist"
  exit 1
fi

export SHERPA_ONNX_RKNN_TOOLKIT2_LIB_DIR=$SHERPA_ONNX_RKNN_TOOLKIT2_PATH/rknpu2/runtime/Linux/librknn_api/aarch64

export CPLUS_INCLUDE_PATH=$SHERPA_ONNX_RKNN_TOOLKIT2_PATH/rknpu2/runtime/Linux/librknn_api/include:$CPLUS_INCLUDE_PATH

if ! command -v aarch64-linux-gnu-gcc  &> /dev/null; then
  echo "Please install a toolchain for cross-compiling."
  echo "You can refer to: "
  echo "  https://k2-fsa.github.io/sherpa/onnx/install/rknn-linux-aarch64.html"
  echo "for help."
  exit 1
fi


dir=$PWD/build-rknn-linux-aarch64
mkdir -p $dir

cd $dir

if [ ! -f alsa-lib/src/.libs/libasound.so ]; then
  echo "Start to cross-compile alsa-lib"
  if [ ! -d alsa-lib ]; then
    git clone --depth 1 --branch v1.2.12 https://github.com/alsa-project/alsa-lib
  fi
  # If it shows:
  #  ./gitcompile: line 79: libtoolize: command not found
  # Please use:
  #  sudo apt-get install libtool m4 automake
  #
  pushd alsa-lib
  CC=aarch64-linux-gnu-gcc ./gitcompile --host=aarch64-linux-gnu
  popd
  echo "Finish cross-compiling alsa-lib"
fi

export CPLUS_INCLUDE_PATH=$PWD/alsa-lib/include:$CPLUS_INCLUDE_PATH
export SHERPA_ONNX_ALSA_LIB_DIR=$PWD/alsa-lib/src/.libs

if [[ x"$BUILD_SHARED_LIBS" == x"" ]]; then
  # By default, use shared link
  BUILD_SHARED_LIBS=ON
fi

cmake \
  -DBUILD_PIPER_PHONMIZE_EXE=OFF \
  -DBUILD_PIPER_PHONMIZE_TESTS=OFF \
  -DBUILD_ESPEAK_NG_EXE=OFF \
  -DBUILD_ESPEAK_NG_TESTS=OFF \
  -DCMAKE_INSTALL_PREFIX=./install \
  -DCMAKE_BUILD_TYPE=Release \
  -DSHERPA_ONNX_ENABLE_GPU=OFF \
  -DBUILD_SHARED_LIBS=$BUILD_SHARED_LIBS \
  -DSHERPA_ONNX_ENABLE_TESTS=OFF \
  -DSHERPA_ONNX_ENABLE_PYTHON=OFF \
  -DSHERPA_ONNX_ENABLE_CHECK=OFF \
  -DSHERPA_ONNX_ENABLE_PORTAUDIO=ON \
  -DSHERPA_ONNX_ENABLE_JNI=OFF \
  -DSHERPA_ONNX_ENABLE_C_API=ON \
  -DSHERPA_ONNX_ENABLE_WEBSOCKET=ON \
  -DSHERPA_ONNX_ENABLE_RKNN=ON \
  -DCMAKE_TOOLCHAIN_FILE=../toolchains/aarch64-linux-gnu.toolchain.cmake \
  ..

make VERBOSE=1 -j4
make install/strip

# Enable it if only needed
# cp -v $SHERPA_ONNX_ALSA_LIB_DIR/libasound.so* ./install/lib/

# See also
# https://github.com/airockchip/rknn-toolkit2/blob/master/rknpu2/examples/rknn_api_demo/build-linux.sh
