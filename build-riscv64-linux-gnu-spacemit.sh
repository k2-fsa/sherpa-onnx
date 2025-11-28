#!/usr/bin/env bash
set -ex

SPACEMIT_TOOLCHAIN=spacemit-toolchain-linux-glibc-x86_64-v1.1.2
DOWNLOAD_URL="https://archive.spacemit.com/toolchain/${SPACEMIT_TOOLCHAIN}.tar.xz"
DOWNLOAD_FILE="./riscv-spacemit-toolchain.tar.gz"

if [ -n "$RISCV_ROOT_PATH" ] && [ -d "$RISCV_ROOT_PATH" ]; then
    echo "LOCAL RISCV_ROOT_PATH: $RISCV_ROOT_PATH"
else
    wget -O "$DOWNLOAD_FILE" "$DOWNLOAD_URL" --quiet --show-progress
    tar -xf "$DOWNLOAD_FILE"
    export RISCV_ROOT_PATH=$PWD/${SPACEMIT_TOOLCHAIN}
    echo "DOWNLOAD RISCV_ROOT_PATH: $RISCV_ROOT_PATH"
fi


if [ x$dir = x"" ]; then
  dir=build-riscv64-linux-gnu-spacemit
fi
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
  CC=$RISCV_ROOT_PATH/bin/riscv64-unknown-linux-gnu-gcc ./gitcompile --host=riscv64-unknown-linux-gnu
  popd
  echo "Finish cross-compiling alsa-lib"
fi

export CPLUS_INCLUDE_PATH=$PWD/alsa-lib/include:$CPLUS_INCLUDE_PATH
export SHERPA_ONNX_ALSA_LIB_DIR=$PWD/alsa-lib/src/.libs

if [[ x"$BUILD_SHARED_LIBS" == x"" ]]; then
  # By default, use shared libraries
  BUILD_SHARED_LIBS=ON
fi

cmake \
  -DBUILD_PIPER_PHONMIZE_EXE=OFF \
  -DBUILD_PIPER_PHONMIZE_TESTS=OFF \
  -DBUILD_ESPEAK_NG_EXE=OFF \
  -DBUILD_ESPEAK_NG_TESTS=OFF \
  -DCMAKE_INSTALL_PREFIX=./install \
  -DCMAKE_BUILD_TYPE=Release \
  -DBUILD_SHARED_LIBS=$BUILD_SHARED_LIBS \
  -DSHERPA_ONNX_ENABLE_TESTS=OFF \
  -DSHERPA_ONNX_ENABLE_PYTHON=OFF \
  -DSHERPA_ONNX_ENABLE_CHECK=OFF \
  -DSHERPA_ONNX_ENABLE_PORTAUDIO=OFF \
  -DSHERPA_ONNX_ENABLE_JNI=OFF \
  -DSHERPA_ONNX_ENABLE_C_API=OFF \
  -DSHERPA_ONNX_ENABLE_WEBSOCKET=ON \
  -DSHERPA_ONNX_ENABLE_SPACEMIT=ON \
  -DSHERPA_ONNX_ENABLE_C_API=ON \
  -DCMAKE_TOOLCHAIN_FILE=../toolchains/riscv64-linux-gnu-spacemit.toolchain.cmake \
  ..

make VERBOSE=1 -j4
make install/strip

# Enable it if only needed
# cp -v $SHERPA_ONNX_ALSA_LIB_DIR/libasound.so* ./install/lib/
