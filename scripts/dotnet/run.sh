#!/usr/bin/env bash
# Copyright (c)  2023  Xiaomi Corporation

set -ex

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
SHERPA_ONNX_DIR=$(cd $SCRIPT_DIR/../.. && pwd)
echo "SCRIPT_DIR: $SCRIPT_DIR"
echo "SHERPA_ONNX_DIR: $SHERPA_ONNX_DIR"

SHERPA_ONNX_VERSION=$(grep "SHERPA_ONNX_VERSION" $SHERPA_ONNX_DIR/CMakeLists.txt  | cut -d " " -f 2  | cut -d '"' -f 2)

mkdir -p /tmp/
pushd /tmp

mkdir -p linux macos windows

# You can pre-download the required wheels to /tmp
src_dir=/tmp

linux_wheel=$src_dir/sherpa_onnx-${SHERPA_ONNX_VERSION}-cp38-cp38-manylinux_2_28_x86_64.whl
macos_wheel=$src_dir/sherpa_onnx-${SHERPA_ONNX_VERSION}-cp38-cp38-macosx_11_0_x86_64.whl
windows_wheel=$src_dir/sherpa_onnx-${SHERPA_ONNX_VERSION}-cp38-cp38-win_amd64.whl

if [ ! -f /tmp/linux/libsherpa-onnx-core.so ]; then
  echo "---linux x86_64---"
  cd linux
  mkdir -p wheel
  cd wheel
  if [ -f $linux_wheel ]; then
    cp -v $linux_wheel .
  else
    curl -OL https://huggingface.co/csukuangfj/sherpa-onnx-wheels/resolve/main/sherpa_onnx-${SHERPA_ONNX_VERSION}-cp38-cp38-manylinux_2_28_x86_64.whl
  fi
  unzip sherpa_onnx-${SHERPA_ONNX_VERSION}-cp38-cp38-manylinux_2_28_x86_64.whl
  cp -v sherpa_onnx/lib/*.so* ../
  cd ..
  rm -v libpiper_phonemize.so libpiper_phonemize.so.1.2.0
  rm -v libsherpa-onnx-fst.so
  rm -v libonnxruntime.so
  rm -v libcargs.so
  rm -rf wheel
  ls -lh
  cd ..
fi

if [ ! -f /tmp/macos/libsherpa-onnx-core.dylib ]; then
  echo "---macOS x86_64---"
  cd macos
  mkdir -p wheel
  cd wheel
  if [ -f $macos_wheel  ]; then
    cp -v $macos_wheel .
  else
    curl -OL https://huggingface.co/csukuangfj/sherpa-onnx-wheels/resolve/main/sherpa_onnx-${SHERPA_ONNX_VERSION}-cp38-cp38-macosx_11_0_x86_64.whl
  fi
  unzip sherpa_onnx-${SHERPA_ONNX_VERSION}-cp38-cp38-macosx_11_0_x86_64.whl
  cp -v sherpa_onnx/lib/*.dylib ../

  cd ..

  rm -v libcargs.dylib
  rm -v libonnxruntime.dylib
  rm -v libpiper_phonemize.1.2.0.dylib libpiper_phonemize.dylib
  rm -v libsherpa-onnx-fst.dylib
  rm -rf wheel
  ls -lh
  cd ..
fi


if [ ! -f /tmp/windows/libsherpa-onnx-core.dll ]; then
  echo "---windows x64---"
  cd windows
  mkdir -p wheel
  cd wheel
  if [ -f $windows_wheel ]; then
    cp -v $windows_wheel .
  else
    curl -OL https://huggingface.co/csukuangfj/sherpa-onnx-wheels/resolve/main/sherpa_onnx-${SHERPA_ONNX_VERSION}-cp38-cp38-win_amd64.whl
  fi
  unzip sherpa_onnx-${SHERPA_ONNX_VERSION}-cp38-cp38-win_amd64.whl
  cp -v sherpa_onnx-${SHERPA_ONNX_VERSION}.data/data/bin/*.dll ../
  cp -v sherpa_onnx-${SHERPA_ONNX_VERSION}.data/data/bin/*.lib ../
  cd ..

  rm -rf wheel
  ls -lh
  cd ..
fi

popd

mkdir -p macos linux windows all

cp ./online.cs all
cp ./offline.cs all

./generate.py

pushd linux
dotnet build -c Release
dotnet pack -c Release -o ../packages
popd

pushd macos
dotnet build -c Release
dotnet pack -c Release -o ../packages
popd

pushd windows
dotnet build -c Release
dotnet pack -c Release -o ../packages
popd

pushd all
dotnet build -c Release
dotnet pack -c Release -o ../packages
popd

ls -lh packages

mkdir -p /tmp/packages
cp -v packages/*.nupkg /tmp/packages
