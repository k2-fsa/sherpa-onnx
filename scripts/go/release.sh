#!/usr/bin/env bash

set -ex

git config --global user.email "csukuangfj@gmail.com"
git config --global user.name "Fangjun Kuang"

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
SHERPA_ONNX_DIR=$(realpath $SCRIPT_DIR/../..)
echo "SCRIPT_DIR: $SCRIPT_DIR"
echo "SHERPA_ONNX_DIR: $SHERPA_ONNX_DIR"


SHERPA_ONNX_VERSION=$(grep "SHERPA_ONNX_VERSION" $SHERPA_ONNX_DIR/CMakeLists.txt  | cut -d " " -f 2  | cut -d '"' -f 2)
echo "SHERPA_ONNX_VERSION $SHERPA_ONNX_VERSION"

function linux() {
  echo "Process linux"
  git clone git@github.com:k2-fsa/sherpa-onnx-go-linux.git
  rm -v ./sherpa-onnx-go-linux/*.go

  cp -v ./sherpa_onnx.go ./sherpa-onnx-go-linux/
  cp -v ./_internal/c-api.h ./sherpa-onnx-go-linux/
  cp -v ./_internal/build_linux_*.go ./sherpa-onnx-go-linux/

  rm -rf sherpa-onnx-go-linux/lib/x86_64-unknown-linux-gnu/lib*
  dst=$(realpath sherpa-onnx-go-linux/lib/x86_64-unknown-linux-gnu)
  mkdir t
  cd t
  wget -q https://huggingface.co/csukuangfj/sherpa-onnx-wheels/resolve/main/sherpa_onnx-${SHERPA_ONNX_VERSION}-cp38-cp38-manylinux_2_17_x86_64.manylinux2014_x86_64.whl
  unzip ./sherpa_onnx-${SHERPA_ONNX_VERSION}-cp38-cp38-manylinux_2_17_x86_64.manylinux2014_x86_64.whl

  cp -v sherpa_onnx/lib/*.so* $dst

  cd ..
  rm -rf t

  rm -rf sherpa-onnx-go-linux/lib/aarch64-unknown-linux-gnu/lib*
  dst=$(realpath sherpa-onnx-go-linux/lib/aarch64-unknown-linux-gnu)
  mkdir t
  cd t
  wget -q https://huggingface.co/csukuangfj/sherpa-onnx-wheels/resolve/main/sherpa_onnx-${SHERPA_ONNX_VERSION}-cp38-cp38-manylinux_2_17_aarch64.manylinux2014_aarch64.whl
  unzip ./sherpa_onnx-${SHERPA_ONNX_VERSION}-cp38-cp38-manylinux_2_17_aarch64.manylinux2014_aarch64.whl

  cp -v sherpa_onnx/lib/*.so* $dst

  cd ..
  rm -rf t

  rm -rf sherpa-onnx-go-linux/lib/arm-unknown-linux-gnueabihf/lib*
  dst=$(realpath sherpa-onnx-go-linux/lib/arm-unknown-linux-gnueabihf)
  mkdir t
  cd t
  wget -q https://huggingface.co/csukuangfj/sherpa-onnx-wheels/resolve/main/sherpa_onnx-${SHERPA_ONNX_VERSION}-cp38-cp38-linux_armv7l.whl
  unzip ./sherpa_onnx-${SHERPA_ONNX_VERSION}-cp38-cp38-linux_armv7l.whl

  cp -v sherpa_onnx/lib/*.so* $dst

  cd ..
  rm -rf t

  echo "------------------------------"
  cd sherpa-onnx-go-linux
  git status
  git add .
  git commit -m "Release v$SHERPA_ONNX_VERSION" && \
  git push && \
  git tag v$SHERPA_ONNX_VERSION && \
  git push origin v$SHERPA_ONNX_VERSION || true
  cd ..
  rm -rf sherpa-onnx-go-linux
}

function osx() {
  echo "Process osx-x64"
  git clone git@github.com:k2-fsa/sherpa-onnx-go-macos.git
  rm -v ./sherpa-onnx-go-macos/*.go
  cp -v ./sherpa_onnx.go ./sherpa-onnx-go-macos/
  cp -v ./_internal/c-api.h ./sherpa-onnx-go-macos/
  cp -v ./_internal/build_darwin_*.go ./sherpa-onnx-go-macos/

  rm -rf sherpa-onnx-go-macos/lib/x86_64-apple-darwin/lib*
  dst=$(realpath sherpa-onnx-go-macos/lib/x86_64-apple-darwin/)

  mkdir t
  cd t
  wget -q https://huggingface.co/csukuangfj/sherpa-onnx-wheels/resolve/main/sherpa_onnx-${SHERPA_ONNX_VERSION}-cp39-cp39-macosx_11_0_x86_64.whl
  unzip ./sherpa_onnx-${SHERPA_ONNX_VERSION}-cp39-cp39-macosx_11_0_x86_64.whl

  cp -v sherpa_onnx/lib/*.dylib $dst/

  pushd $dst
  cp -v libonnxruntime.1.18.0.dylib libonnxruntime.dylib
  popd

  cd ..
  rm -rf t

  echo "process macos arm64"
  rm -rf sherpa-onnx-go-macos/lib/aarch64-apple-darwin/lib*
  dst=$(realpath sherpa-onnx-go-macos/lib/aarch64-apple-darwin)

  mkdir t
  cd t
  wget -q https://huggingface.co/csukuangfj/sherpa-onnx-wheels/resolve/main/sherpa_onnx-${SHERPA_ONNX_VERSION}-cp39-cp39-macosx_11_0_arm64.whl
  unzip ./sherpa_onnx-${SHERPA_ONNX_VERSION}-cp39-cp39-macosx_11_0_arm64.whl

  cp -v sherpa_onnx/lib/*.dylib $dst/

  pushd $dst
  cp -v libonnxruntime.1.18.0.dylib libonnxruntime.dylib
  popd

  cd ..
  rm -rf t
  echo "------------------------------"
  cd sherpa-onnx-go-macos
  git status
  git add .
  git commit -m "Release v$SHERPA_ONNX_VERSION" && \
  git push && \
  git tag v$SHERPA_ONNX_VERSION && \
  git push origin v$SHERPA_ONNX_VERSION || true
  cd ..
  rm -rf sherpa-onnx-go-macos
}

function windows() {
  echo "Process windows"
  git clone git@github.com:k2-fsa/sherpa-onnx-go-windows.git
  rm -v ./sherpa-onnx-go-windows/*.go
  cp -v ./sherpa_onnx.go ./sherpa-onnx-go-windows/
  cp -v ./_internal/c-api.h ./sherpa-onnx-go-windows/
  cp -v ./_internal/build_windows_*.go ./sherpa-onnx-go-windows/

  rm -fv sherpa-onnx-go-windows/lib/x86_64-pc-windows-gnu/*
  dst=$(realpath sherpa-onnx-go-windows/lib/x86_64-pc-windows-gnu)
  mkdir t
  cd t
  wget -q https://huggingface.co/csukuangfj/sherpa-onnx-wheels/resolve/main/sherpa_onnx-${SHERPA_ONNX_VERSION}-cp38-cp38-win_amd64.whl
  unzip ./sherpa_onnx-${SHERPA_ONNX_VERSION}-cp38-cp38-win_amd64.whl

  cp -v sherpa_onnx-${SHERPA_ONNX_VERSION}.data/data/bin/*.dll $dst

  cd ..
  rm -rf t

  rm -fv sherpa-onnx-go-windows/lib/i686-pc-windows-gnu/*
  dst=$(realpath sherpa-onnx-go-windows/lib/i686-pc-windows-gnu)
  mkdir t
  cd t
  wget -q https://huggingface.co/csukuangfj/sherpa-onnx-wheels/resolve/main/sherpa_onnx-${SHERPA_ONNX_VERSION}-cp38-cp38-win32.whl
  unzip ./sherpa_onnx-${SHERPA_ONNX_VERSION}-cp38-cp38-win32.whl

  cp -v sherpa_onnx-${SHERPA_ONNX_VERSION}.data/data/bin/*.dll $dst

  cd ..
  rm -rf t
  echo "------------------------------"
  cd sherpa-onnx-go-windows
  git status
  git add .
  git commit -m "Release v$SHERPA_ONNX_VERSION" && \
  git push && \
  git tag v$SHERPA_ONNX_VERSION && \
  git push origin v$SHERPA_ONNX_VERSION || true
  cd ..
  rm -rf sherpa-onnx-go-windows
}


windows
linux
osx

rm -fv ~/.ssh/github
