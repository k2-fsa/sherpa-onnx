#!/usr/bin/env bash
set -ex

if ! command -v riscv64-unknown-linux-gnu-g++  &> /dev/null; then
  echo "Please install the toolchain first."
  echo
  echo "You can use the following command to install the toolchain:"
  echo
  echo "  mkdir -p $HOME/software"
  echo "  cd $HOME/software"
  echo "  mkdir riscv64-glibc-ubuntu-18.04-nightly-2022.11.12-nightly"
  echo "  cd riscv64-glibc-ubuntu-18.04-nightly-2022.11.12-nightly"
  echo "  wget https://github.com/riscv-collab/riscv-gnu-toolchain/releases/download/2022.11.12/riscv64-glibc-ubuntu-18.04-nightly-2022.11.12-nightly.tar.gz"
  echo "  tar xvf riscv64-glibc-ubuntu-18.04-nightly-2022.11.12-nightly.tar.gz --strip-components=1"
  echo "  export PATH=$HOME/software/riscv64-glibc-ubuntu-18.04-nightly-2022.11.12-nightly/bin"
  echo
  exit 1
fi

if [ x$dir = x"" ]; then
  dir=build-riscv64-linux-gnu
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
  CC=riscv64-unknown-linux-gnu-gcc ./gitcompile --host=riscv64-unknown-linux-gnu
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
  -DCMAKE_TOOLCHAIN_FILE=../toolchains/riscv64-linux-gnu.toolchain.cmake \
  ..

make VERBOSE=1 -j4
make install/strip

# Enable it if only needed
# cp -v $SHERPA_ONNX_ALSA_LIB_DIR/libasound.so* ./install/lib/
