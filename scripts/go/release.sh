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
  echo "Process linux-x64"
  git clone git@github.com:k2-fsa/sherpa-onnx-go-linux.git
  cp -v ./sherpa_onnx.go ./sherpa-onnx-go-linux/
  cp -v ./_internal/c-api.h ./sherpa-onnx-go-linux/

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
  git commit -m "Release $SHERPA_ONNX_VERSION" && \
  git push && \
  git tag $SHERPA_ONNX_VERSION && \
  git push origin $SHERPA_ONNX_VERSION || true
  cd ..
  rm -rf sherpa-onnx-go-linux
}

function osx() {
  echo "Process osx-x64"
  git clone git@github.com:k2-fsa/sherpa-onnx-go-macos.git
  cp -v ./sherpa_onnx.go ./sherpa-onnx-go-macos/
  cp -v ./_internal/c-api.h ./sherpa-onnx-go-macos/

  rm -rf sherpa-onnx-go-macos/lib/x86_64-apple-darwin/lib*
  dst=$(realpath sherpa-onnx-go-macos/lib/x86_64-apple-darwin/)

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

  echo "process macos arm64"
  rm -rf sherpa-onnx-go-macos/lib/aarch64-apple-darwin/lib*
  dst=$(realpath sherpa-onnx-go-macos/lib/aarch64-apple-darwin)

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
  echo "------------------------------"
  cd sherpa-onnx-go-macos
  git status
  git add .
  git commit -m "Release $SHERPA_ONNX_VERSION" && \
  git push && \
  git tag $SHERPA_ONNX_VERSION && \
  git push origin $SHERPA_ONNX_VERSION || true
  cd ..
  rm -rf sherpa-onnx-go-macos
}


linux
osx







if false; then



echo "========================================================================="

git clone git@github.com:k2-fsa/sherpa-onnx-go-linux.git

echo "Copy libs for Linux x86_64"

rm -rf sherpa-onnx-go-linux/lib/x86_64-unknown-linux-gnu/lib*

cp -v ./linux/sherpa_onnx/lib/libkaldi-native-fbank-core.so sherpa-onnx-go-linux/lib/x86_64-unknown-linux-gnu/
cp -v ./linux/sherpa_onnx/lib/libonnxruntime* sherpa-onnx-go-linux/lib/x86_64-unknown-linux-gnu/
cp -v ./linux/sherpa_onnx/lib/libsherpa-onnx-c-api.so sherpa-onnx-go-linux/lib/x86_64-unknown-linux-gnu/
cp -v ./linux/sherpa_onnx/lib/libsherpa-onnx-core.so sherpa-onnx-go-linux/lib/x86_64-unknown-linux-gnu/

echo "Copy sources for Linux x86_64"
cp sherpa-onnx/c-api/c-api.h sherpa-onnx-go-linux/
cp scripts/go/sherpa_onnx.go sherpa-onnx-go-linux/

pushd sherpa-onnx-go-linux
tag=$(git describe --abbrev=0 --tags)
if [[ x"$VERSION" == x"auto" ]]; then
  # this is a pre-release
  if [[ $tag == ${SHERPA_ONNX_VERSION}* ]]; then
    # echo we have already release pre-release before, so just increment it
    last=$(echo $tag | rev | cut -d'.' -f 1 | rev)
    new_last=$((last+1))
    new_tag=${SHERPA_ONNX_VERSION}-alpha.${new_last}
  else
    new_tag=${SHERPA_ONNX_VERSION}-alpha.1
  fi
else
  new_tag=$VERSION
fi

echo "new_tag: $new_tag"
git add .
git status
git commit -m "Release $new_tag" && \
git push && \
git tag $new_tag && \
git push origin $new_tag || true

popd
echo "========================================================================="

git clone git@github.com:k2-fsa/sherpa-onnx-go-macos.git

echo "Copy libs for macOS x86_64"
rm -rf sherpa-onnx-go-macos/lib/x86_64-apple-darwin/lib*
cp -v ./macos-x86_64/libkaldi-native-fbank-core.dylib sherpa-onnx-go-macos/lib/x86_64-apple-darwin
cp -v ./macos-x86_64/libonnxruntime* sherpa-onnx-go-macos/lib/x86_64-apple-darwin
cp -v ./macos-x86_64/libsherpa-onnx-c-api.dylib sherpa-onnx-go-macos/lib/x86_64-apple-darwin
cp -v ./macos-x86_64/libsherpa-onnx-core.dylib sherpa-onnx-go-macos/lib/x86_64-apple-darwin

echo "Copy libs for macOS arm64"
rm -rf sherpa-onnx-go-macos/lib/aarch64-apple-darwin/lib*
cp -v ./macos-arm64/libkaldi-native-fbank-core.dylib sherpa-onnx-go-macos/lib/aarch64-apple-darwin
cp -v ./macos-arm64/libonnxruntime* sherpa-onnx-go-macos/lib/aarch64-apple-darwin
cp -v ./macos-arm64/libsherpa-onnx-c-api.dylib sherpa-onnx-go-macos/lib/aarch64-apple-darwin
cp -v ./macos-arm64/libsherpa-onnx-core.dylib sherpa-onnx-go-macos/lib/aarch64-apple-darwin

echo "Copy sources for macOS"
cp sherpa-onnx/c-api/c-api.h sherpa-onnx-go-macos/
cp scripts/go/sherpa_onnx.go sherpa-onnx-go-macos/

pushd sherpa-onnx-go-macos
tag=$(git describe --abbrev=0 --tags)
if [[ x"$VERSION" == x"auto" ]]; then
  # this is a pre-release
  if [[ $tag == ${SHERPA_ONNX_VERSION}* ]]; then
    # echo we have already release pre-release before, so just increment it
    last=$(echo $tag | rev | cut -d'.' -f 1 | rev)
    new_last=$((last+1))
    new_tag=${SHERPA_ONNX_VERSION}-alpha.${new_last}
  else
    new_tag=${SHERPA_ONNX_VERSION}-alpha.1
  fi
else
  new_tag=$VERSION
fi

echo "new_tag: $new_tag"
git add .
git status
git commit -m "Release $new_tag" && \
git push && \
git tag $new_tag && \
git push origin $new_tag || true

popd
echo "========================================================================="

git clone git@github.com:k2-fsa/sherpa-onnx-go-windows.git
echo "Copy libs for Windows x86_64"
rm -fv sherpa-onnx-go-windows/lib/x86_64-pc-windows-gnu/*
cp -v ./windows-x64/kaldi-native-fbank-core.dll sherpa-onnx-go-windows/lib/x86_64-pc-windows-gnu
cp -v ./windows-x64/onnxruntime.dll sherpa-onnx-go-windows/lib/x86_64-pc-windows-gnu
cp -v ./windows-x64/sherpa-onnx-c-api.dll sherpa-onnx-go-windows/lib/x86_64-pc-windows-gnu
cp -v ./windows-x64/sherpa-onnx-core.dll sherpa-onnx-go-windows/lib/x86_64-pc-windows-gnu

echo "Copy libs for Windows x86"
rm -fv sherpa-onnx-go-windows/lib/i686-pc-windows-gnu/*
cp -v ./windows-win32/kaldi-native-fbank-core.dll sherpa-onnx-go-windows/lib/i686-pc-windows-gnu
cp -v ./windows-win32/onnxruntime.dll sherpa-onnx-go-windows/lib/i686-pc-windows-gnu
cp -v ./windows-win32/sherpa-onnx-c-api.dll sherpa-onnx-go-windows/lib/i686-pc-windows-gnu
cp -v ./windows-win32/sherpa-onnx-core.dll sherpa-onnx-go-windows/lib/i686-pc-windows-gnu

echo "Copy sources for Windows"
cp sherpa-onnx/c-api/c-api.h sherpa-onnx-go-windows/
cp scripts/go/sherpa_onnx.go sherpa-onnx-go-windows/

pushd sherpa-onnx-go-windows
tag=$(git describe --abbrev=0 --tags)
if [[ x"$VERSION" == x"auto" ]]; then
  # this is a pre-release
  if [[ $tag == ${SHERPA_ONNX_VERSION}* ]]; then
    # echo we have already release pre-release before, so just increment it
    last=$(echo $tag | rev | cut -d'.' -f 1 | rev)
    new_last=$((last+1))
    new_tag=${SHERPA_ONNX_VERSION}-alpha.${new_last}
  else
    new_tag=${SHERPA_ONNX_VERSION}-alpha.1
  fi
else
  new_tag=$VERSION
fi

echo "new_tag: $new_tag"
git add .
git status
git commit -m "Release $new_tag" && \
git push && \
git tag $new_tag && \
git push origin $new_tag || true

popd

echo "========================================================================="
fi

rm -fv ~/.ssh/github

