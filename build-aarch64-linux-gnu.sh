#!/usr/bin/env bash
#
# Usage of this file
#
# (1) Build CPU version of sherpa-onnx
#    ./build-aarch64-linux-gnu.sh
#
# (2) Build GPU version of sherpa-onnx
#
#   (a) Make sure your board has NVIDIA GPU(s)
#
#   (b) For Jetson Nano B01 (using CUDA 10.2)
#
#       export SHERPA_ONNX_ENABLE_GPU=ON
#       export SHERPA_ONNX_LINUX_ARM64_GPU_ONNXRUNTIME_VERSION=1.11.0
#       ./build-aarch64-linux-gnu.sh
#
#   (c) For Jetson Orin NX (using CUDA 11.4)
#
#       export SHERPA_ONNX_ENABLE_GPU=ON
#       export SHERPA_ONNX_LINUX_ARM64_GPU_ONNXRUNTIME_VERSION=1.16.0
#       ./build-aarch64-linux-gnu.sh

if command -v aarch64-none-linux-gnu-gcc  &> /dev/null; then
  ln -svf $(which aarch64-none-linux-gnu-gcc) ./aarch64-linux-gnu-gcc
  ln -svf $(which aarch64-none-linux-gnu-g++) ./aarch64-linux-gnu-g++
  export PATH=$PWD:$PATH
fi

if ! command -v aarch64-linux-gnu-gcc  &> /dev/null; then
  echo "Please install a toolchain for cross-compiling."
  echo "You can refer to: "
  echo "  https://k2-fsa.github.io/sherpa/onnx/install/aarch64-embedded-linux.html"
  echo "for help."
  exit 1
fi

set -ex

dir=build-aarch64-linux-gnu
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
  # By default, use static link
  BUILD_SHARED_LIBS=OFF
fi

if [[ x"$SHERPA_ONNX_ENABLE_GPU" == x"" ]]; then
  # By default, use CPU
  SHERPA_ONNX_ENABLE_GPU=OFF
fi

if [[ x"$SHERPA_ONNX_ENABLE_GPU" == x"ON" ]]; then
  # Build shared libs if building GPU is enabled.
  BUILD_SHARED_LIBS=ON
fi

if [[ x"$SHERPA_ONNX_LINUX_ARM64_GPU_ONNXRUNTIME_VERSION" == x"" ]]; then
  # Used only when SHERPA_ONNX_ENABLE_GPU is ON
  SHERPA_ONNX_LINUX_ARM64_GPU_ONNXRUNTIME_VERSION="1.11.0"
fi

cmake \
  -DBUILD_PIPER_PHONMIZE_EXE=OFF \
  -DBUILD_PIPER_PHONMIZE_TESTS=OFF \
  -DBUILD_ESPEAK_NG_EXE=OFF \
  -DBUILD_ESPEAK_NG_TESTS=OFF \
  -DCMAKE_INSTALL_PREFIX=./install \
  -DCMAKE_BUILD_TYPE=Release \
  -DSHERPA_ONNX_ENABLE_GPU=$SHERPA_ONNX_ENABLE_GPU \
  -DBUILD_SHARED_LIBS=$BUILD_SHARED_LIBS \
  -DSHERPA_ONNX_ENABLE_TESTS=OFF \
  -DSHERPA_ONNX_ENABLE_PYTHON=OFF \
  -DSHERPA_ONNX_ENABLE_CHECK=OFF \
  -DSHERPA_ONNX_ENABLE_PORTAUDIO=OFF \
  -DSHERPA_ONNX_ENABLE_JNI=OFF \
  -DSHERPA_ONNX_ENABLE_C_API=ON \
  -DSHERPA_ONNX_ENABLE_WEBSOCKET=ON \
  -DSHERPA_ONNX_LINUX_ARM64_GPU_ONNXRUNTIME_VERSION=$SHERPA_ONNX_LINUX_ARM64_GPU_ONNXRUNTIME_VERSION \
  -DCMAKE_TOOLCHAIN_FILE=../toolchains/aarch64-linux-gnu.toolchain.cmake \
  ..

make VERBOSE=1 -j4
make install/strip

# Enable it if only needed
# cp -v $SHERPA_ONNX_ALSA_LIB_DIR/libasound.so* ./install/lib/
