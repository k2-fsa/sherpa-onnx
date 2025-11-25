#!/usr/bin/env bash
#
# Usage of this file
#
# Refer to the "How to build static libraries and statically linked binaries"
# section at:
#   https://k2-fsa.github.io/sherpa/onnx/install/aarch64-embedded-linux.html
#
# Use the following toolchain:
#   aarch64-none-linux-gnu-gcc
#   (GNU Toolchain for the A-profile Architecture 10.3-2021.07 (arm-10.29))
#   version 10.3.1 20210621
#
# Note: Do NOT set:
#   export BUILD_SHARED_LIBS=OFF
#
# Usage of this file
#

set -ex

# Before you run this file, make sure you have first cloned
# https://github.com/AXERA-TECH/ax650n_bsp_sdk
# and set the environment variable SHERPA_ONNX_AXERA_PATH

if [ -z "$AXERA_SDK_ROOT" ]; then
  AXERA_SDK_ROOT=/home/m5stack/Workspace/kaldi/sherpa-onnx/ax650n_bsp_sdk/out
  echo "Please set AXERA_SDK_ROOT to your Axera SDK path, e.g.:"
  echo "  export AXERA_SDK_ROOT=$PWD/ax650n_bsp_sdk/out"
  exit 1
fi

if [ ! -d "$AXERA_SDK_ROOT" ]; then
  echo "AXERA_SDK_ROOT ($AXERA_SDK_ROOT) does not exist"
  exit 1
fi

if [ ! -f "$AXERA_SDK_ROOT/include/ax_engine_api.h" ]; then
  echo "$AXERA_SDK_ROOT/include/ax_engine_api.h does not exist"
  exit 1
fi

if [ ! -f "$AXERA_SDK_ROOT/lib/libax_engine.so" ]; then
  echo "$AXERA_SDK_ROOT/lib/libax_engine.so does not exist"
  exit 1
fi

export CPLUS_INCLUDE_PATH="$AXERA_SDK_ROOT/include:$CPLUS_INCLUDE_PATH"

export SHERPA_ONNX_AXERA_LIB_DIR="$AXERA_SDK_ROOT/lib"

if command -v aarch64-none-linux-gnu-gcc  &> /dev/null; then
  ln -svf $(which aarch64-none-linux-gnu-gcc) ./aarch64-linux-gnu-gcc
  ln -svf $(which aarch64-none-linux-gnu-g++) ./aarch64-linux-gnu-g++
  export PATH=$PWD:$PATH
fi

if ! command -v aarch64-none-linux-gnu-gcc  &> /dev/null; then
  echo "Please install a toolchain for cross-compiling."
  echo "You can refer to: "
  echo "  https://k2-fsa.github.io/sherpa/onnx/install/aarch64-embedded-linux.html"
  echo "for help."
  exit 1
fi


dir=$PWD/build-axera-linux-aarch64
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
  -DALSA_INCLUDE_DIR=$PWD/alsa-lib/include \
  -DALSA_LIBRARY=$PWD/alsa-lib/src/.libs/libasound.so \
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
  -DSHERPA_ONNX_ENABLE_AXERA=ON \
  -DCMAKE_TOOLCHAIN_FILE=../toolchains/aarch64-linux-gnu.toolchain.cmake \
  ..

make VERBOSE=1 -j22
make install/strip

# Enable it if only needed
# cp -v $SHERPA_ONNX_ALSA_LIB_DIR/libasound.so* ./install/lib/

# See also
# https://github.com/airockchip/rknn-toolkit2/blob/master/rknpu2/examples/rknn_api_demo/build-linux.sh
