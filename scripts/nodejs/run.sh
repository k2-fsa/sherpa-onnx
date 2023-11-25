#!/usr/bin/env bash
set -ex

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
SHERPA_ONNX_DIR=$(realpath $SCRIPT_DIR/../..)
echo "SCRIPT_DIR: $SCRIPT_DIR"
echo "SHERPA_ONNX_DIR: $SHERPA_ONNX_DIR"

SHERPA_ONNX_VERSION=$(grep "SHERPA_ONNX_VERSION" $SHERPA_ONNX_DIR/CMakeLists.txt  | cut -d " " -f 2  | cut -d '"' -f 2)

echo "SHERPA_ONNX_VERSION $SHERPA_ONNX_VERSION"
sed -i.bak s/SHERPA_ONNX_VERSION/$SHERPA_ONNX_VERSION/g ./package.json.in

cp package.json.in package.json
rm package.json.in
rm package.json.in.bak
rm .clang-format

function windows_x64() {
  echo "Process Windows (x64)"
  mkdir -p lib/win-x64
  dst=$(realpath lib/win-x64)
  mkdir t
  cd t
  wget -q https://huggingface.co/csukuangfj/sherpa-onnx-wheels/resolve/main/sherpa_onnx-${SHERPA_ONNX_VERSION}-cp38-cp38-win_amd64.whl
  unzip ./sherpa_onnx-${SHERPA_ONNX_VERSION}-cp38-cp38-win_amd64.whl

  cp -v sherpa_onnx-${SHERPA_ONNX_VERSION}.data/data/bin/*.dll $dst
  cp -v sherpa_onnx-${SHERPA_ONNX_VERSION}.data/data/bin/*.lib $dst
  rm -fv $dst/sherpa-onnx-portaudio.dll

  cd ..
  rm -rf t
}

function windows_x86() {
  echo "Process Windows (x86)"
  mkdir -p lib/win-x86
  dst=$(realpath lib/win-x86)
  mkdir t
  cd t
  wget -q https://huggingface.co/csukuangfj/sherpa-onnx-wheels/resolve/main/sherpa_onnx-${SHERPA_ONNX_VERSION}-cp38-cp38-win32.whl
  unzip ./sherpa_onnx-${SHERPA_ONNX_VERSION}-cp38-cp38-win32.whl

  cp -v sherpa_onnx-${SHERPA_ONNX_VERSION}.data/data/bin/*.dll $dst
  cp -v sherpa_onnx-${SHERPA_ONNX_VERSION}.data/data/bin/*.lib $dst
  rm -fv $dst/sherpa-onnx-portaudio.dll

  cd ..
  rm -rf t
}

function linux_x64() {
  echo "Process Linux (x64)"
  mkdir -p lib/linux-x64
  dst=$(realpath lib/linux-x64)
  mkdir t
  cd t
  wget -q https://huggingface.co/csukuangfj/sherpa-onnx-wheels/resolve/main/sherpa_onnx-${SHERPA_ONNX_VERSION}-cp38-cp38-manylinux_2_17_x86_64.manylinux2014_x86_64.whl
  unzip ./sherpa_onnx-${SHERPA_ONNX_VERSION}-cp38-cp38-manylinux_2_17_x86_64.manylinux2014_x86_64.whl

  cp -v sherpa_onnx/lib/*.so* $dst
  rm -v $dst/libcargs.so
  rm -v $dst/libsherpa-onnx-portaudio.so
  rm -v $dst/libsherpa-onnx-fst.so
  rm -v $dst/libonnxruntime.so

  cd ..
  rm -rf t
}

function osx_x64() {
  echo "Process osx-x64"
  mkdir -p lib/osx-x64
  dst=$(realpath lib/osx-x64)
  mkdir t
  cd t
  wget -q https://huggingface.co/csukuangfj/sherpa-onnx-wheels/resolve/main/sherpa_onnx-${SHERPA_ONNX_VERSION}-cp38-cp38-macosx_10_14_x86_64.whl
  unzip ./sherpa_onnx-${SHERPA_ONNX_VERSION}-cp38-cp38-macosx_10_14_x86_64.whl

  cp -v sherpa_onnx/lib/*.dylib $dst/
  rm -v $dst/libonnxruntime.dylib
  rm -v $dst/libcargs.dylib
  rm -v $dst/libsherpa-onnx-fst.dylib
  rm -v $dst/libsherpa-onnx-portaudio.dylib

  cd ..
  rm -rf t
}

function osx_arm64() {
  echo "Process osx-arm64"
  mkdir -p lib/osx-arm64
  dst=$(realpath lib/osx-arm64)
  mkdir t
  cd t
  wget -q https://huggingface.co/csukuangfj/sherpa-onnx-wheels/resolve/main/sherpa_onnx-${SHERPA_ONNX_VERSION}-cp38-cp38-macosx_11_0_arm64.whl
  unzip ./sherpa_onnx-${SHERPA_ONNX_VERSION}-cp38-cp38-macosx_11_0_arm64.whl

  cp -v sherpa_onnx/lib/*.dylib $dst/
  rm -v $dst/libonnxruntime.dylib
  rm -v $dst/libcargs.dylib
  rm -v $dst/libsherpa-onnx-fst.dylib
  rm -v $dst/libsherpa-onnx-portaudio.dylib

  cd ..
  rm -rf t
}

windows_x64
ls -lh lib/win-x64

windows_x86
ls -lh lib/win-x86

linux_x64
ls -lh lib/linux-x64

osx_x64
ls -lh lib/osx-x64

osx_arm64
ls -lh lib/osx-arm64
