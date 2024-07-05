#!/usr/bin/env bash
# Copyright (c)  2023  Xiaomi Corporation

set -ex

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
SHERPA_ONNX_DIR=$(cd $SCRIPT_DIR/../.. && pwd)
echo "SCRIPT_DIR: $SCRIPT_DIR"
echo "SHERPA_ONNX_DIR: $SHERPA_ONNX_DIR"

SHERPA_ONNX_VERSION=$(grep "SHERPA_ONNX_VERSION" $SHERPA_ONNX_DIR/CMakeLists.txt  | cut -d " " -f 2  | cut -d '"' -f 2)

# You can pre-download the required wheels to $src_dir

if [ $(hostname) == fangjuns-MacBook-Pro.local ]; then
  HF_MIRROR=hf-mirror.com
  src_dir=/Users/fangjun/open-source/sherpa-onnx/scripts/dotnet/tmp
else
  src_dir=/tmp
  HF_MIRROR=hf.co
fi
export src_dir

mkdir -p $src_dir
pushd $src_dir

mkdir -p linux macos-x64 macos-arm64 windows-x64 windows-x86 windows-arm64

linux_wheel_filename=sherpa_onnx-${SHERPA_ONNX_VERSION}-cp38-cp38-manylinux_2_17_x86_64.manylinux2014_x86_64.whl
linux_wheel=$src_dir/$linux_wheel_filename

macos_x64_wheel_filename=sherpa_onnx-${SHERPA_ONNX_VERSION}-cp39-cp39-macosx_11_0_x86_64.whl
macos_x64_wheel=$src_dir/$macos_x64_wheel_filename

macos_arm64_wheel_filename=sherpa_onnx-${SHERPA_ONNX_VERSION}-cp39-cp39-macosx_11_0_arm64.whl
macos_arm64_wheel=$src_dir/$macos_arm64_wheel_filename

windows_x64_wheel_filename=sherpa_onnx-${SHERPA_ONNX_VERSION}-cp38-cp38-win_amd64.whl
windows_x64_wheel=$src_dir/$windows_x64_wheel_filename

windows_x86_wheel_filename=sherpa-onnx-${SHERPA_ONNX_VERSION}-win-x86.tar.bz2
windows_x86_wheel=$src_dir/$windows_x86_wheel_filename

windows_arm64_wheel_filename=sherpa-onnx-${SHERPA_ONNX_VERSION}-win-arm64.tar.bz2
windows_arm64_wheel=$src_dir/$windows_arm64_wheel_filename

if [ ! -f $src_dir/linux/libsherpa-onnx-c-api.so ]; then
  echo "---linux x86_64---"
  cd linux
  mkdir -p wheel
  cd wheel
  if [ -f $linux_wheel ]; then
    cp -v $linux_wheel .
  else
    curl -OL https://$HF_MIRROR/csukuangfj/sherpa-onnx-wheels/resolve/main/$linux_wheel_filename
  fi
  unzip $linux_wheel_filename
  cp -v sherpa_onnx/lib/*.so* ../
  cd ..
  rm -rf wheel
  ls -lh
  cd ..
fi

if [ ! -f $src_dir/macos-x64/libsherpa-onnx-c-api.dylib ]; then
  echo "--- macOS x86_64---"
  cd macos-x64
  mkdir -p wheel
  cd wheel
  if [ -f $macos_x64_wheel  ]; then
    cp -v $macos_x64_wheel .
  else
    curl -OL https://$HF_MIRROR/csukuangfj/sherpa-onnx-wheels/resolve/main/$macos_x64_wheel_filename
  fi
  unzip $macos_x64_wheel_filename
  cp -v sherpa_onnx/lib/*.dylib ../

  cd ..

  rm -rf wheel
  ls -lh
  cd ..
fi

if [ ! -f $src_dir/macos-arm64/libsherpa-onnx-c-api.dylib ]; then
  echo "--- macOS arm64---"
  cd macos-arm64
  mkdir -p wheel
  cd wheel
  if [ -f $macos_arm64_wheel  ]; then
    cp -v $macos_arm64_wheel .
  else
    curl -OL https://$HF_MIRROR/csukuangfj/sherpa-onnx-wheels/resolve/main/$macos_arm64_wheel_filename
  fi
  unzip $macos_arm64_wheel_filename
  cp -v sherpa_onnx/lib/*.dylib ../

  cd ..

  rm -rf wheel
  ls -lh
  cd ..
fi

if [ ! -f $src_dir/windows-x64/sherpa-onnx-c-api.dll ]; then
  echo "---windows x64---"
  cd windows-x64
  mkdir -p wheel
  cd wheel
  if [ -f $windows_x64_wheel ]; then
    cp -v $windows_x64_wheel .
  else
    curl -OL https://$HF_MIRROR/csukuangfj/sherpa-onnx-wheels/resolve/main/$windows_x64_wheel_filename
  fi
  unzip $windows_x64_wheel_filename
  cp -v sherpa_onnx-${SHERPA_ONNX_VERSION}.data/data/bin/*.dll ../
  cd ..

  rm -rf wheel
  ls -lh
  cd ..
fi

if [ ! -f $src_dir/windows-x86/sherpa-onnx-c-api.dll ]; then
  echo "---windows x86---"
  cd windows-x86
  mkdir -p wheel
  cd wheel
  if [ -f $windows_x86_wheel ]; then
    cp -v $windows_x86_wheel .
  else
    curl -OL https://$HF_MIRROR/csukuangfj/sherpa-onnx-libs/resolve/main/windows-for-dotnet/$windows_x86_wheel_filename
  fi
  tar xvf $windows_x86_wheel_filename
  cp -v sherpa-onnx-${SHERPA_ONNX_VERSION}-win-x86/*dll ../
  cd ..

  rm -rf wheel
  ls -lh
  cd ..
fi

if [ ! -f $src_dir/windows-arm64/sherpa-onnx-c-api.dll ]; then
  echo "---windows arm64---"
  cd windows-arm64
  mkdir -p wheel
  cd wheel
  if [ -f $windows_arm64_wheel ]; then
    cp -v $windows_arm64_wheel .
  else
    curl -OL https://$HF_MIRROR/csukuangfj/sherpa-onnx-libs/resolve/main/windows-for-dotnet/$windows_arm64_wheel_filename
  fi
  tar xvf $windows_arm64_wheel_filename
  cp -v sherpa-onnx-${SHERPA_ONNX_VERSION}-win-arm64/*dll ../
  cd ..

  rm -rf wheel
  ls -lh
  cd ..
fi

popd

mkdir -p macos-x64 macos-arm64 linux windows-x64 windows-x86 windows-arm64 all

cp ./*.cs all

./generate.py

pushd linux
dotnet build -c Release
dotnet pack -c Release -o ../packages
popd

pushd macos-x64
dotnet build -c Release
dotnet pack -c Release -o ../packages
popd

pushd macos-arm64
dotnet build -c Release
dotnet pack -c Release -o ../packages
popd

pushd windows-x64
dotnet build -c Release
dotnet pack -c Release -o ../packages
popd

pushd windows-x86
dotnet build -c Release
dotnet pack -c Release -o ../packages
popd

pushd windows-arm64
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
