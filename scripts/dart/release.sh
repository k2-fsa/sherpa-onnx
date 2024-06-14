#!/usr/bin/env bash

# see
# https://dart.dev/tools/pub/automated-publishing

set -ex

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
SHERPA_ONNX_DIR=$(cd $SCRIPT_DIR/../.. && pwd)
echo "SCRIPT_DIR: $SCRIPT_DIR"
echo "SHERPA_ONNX_DIR: $SHERPA_ONNX_DIR"

SHERPA_ONNX_VERSION=$(grep "SHERPA_ONNX_VERSION" $SHERPA_ONNX_DIR/CMakeLists.txt  | cut -d " " -f 2  | cut -d '"' -f 2)

src_dir=$SHERPA_ONNX_DIR/sherpa-onnx/flutter
pushd $src_dir

v="version: $SHERPA_ONNX_VERSION"
echo "v: $v"
sed -i.bak s"/^version: .*/$v/" ./pubspec.yaml
rm *.bak
rm notes.md
git status
git diff

HF_MIRROR=hf.co
linux_wheel_filename=sherpa_onnx-${SHERPA_ONNX_VERSION}-cp38-cp38-manylinux_2_17_x86_64.manylinux2014_x86_64.whl
linux_wheel=$src_dir/$linux_wheel_filename

macos_wheel_filename=sherpa_onnx-${SHERPA_ONNX_VERSION}-cp39-cp39-macosx_10_14_universal2.whl
macos_wheel=$src_dir/$macos_wheel_filename

windows_x64_wheel_filename=sherpa_onnx-${SHERPA_ONNX_VERSION}-cp38-cp38-win_amd64.whl
windows_x64_wheel=$src_dir/$windows_x64_wheel_filename

function process_linux() {
  mkdir -p t
  cd t
  curl -OL https://$HF_MIRROR/csukuangfj/sherpa-onnx-wheels/resolve/main/$linux_wheel_filename
  unzip $linux_wheel_filename
  cp -v sherpa_onnx/lib/*.so* ../linux
  cd ..
  rm -rf t

  pushd linux

  rm -v libpiper_phonemize.so libpiper_phonemize.so.1.2.0
  rm -v libonnxruntime.so
  rm -v libcargs.so

  popd
}

function process_windows_x64() {
  mkdir -p t
  cd t
  curl -OL https://$HF_MIRROR/csukuangfj/sherpa-onnx-wheels/resolve/main/$windows_x64_wheel_filename
  unzip $windows_x64_wheel_filename
  cp -v sherpa_onnx-${SHERPA_ONNX_VERSION}.data/data/bin/*.dll ../windows
  cd ..
  rm -rf t
}

function process_macos() {
  mkdir -p t
  cd t
  curl -OL https://$HF_MIRROR/csukuangfj/sherpa-onnx-wheels/resolve/main/$macos_wheel_filename
  unzip $macos_wheel_filename
  cp -v sherpa_onnx/lib/*.dylib ../macos
  cd ..
  rm -rf t

  pushd macos
  rm -v libcargs.dylib
  rm -v libonnxruntime.dylib
  rm -v libpiper_phonemize.1.2.0.dylib libpiper_phonemize.dylib
  popd
}

process_linux
process_windows_x64
process_macos
