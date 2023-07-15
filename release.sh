#!/usr/bin/env bash
#
# Copyright (c)  2023  Xiaomi Corporation
#
# Please see the end of this file for what files it will generate

set -ex
SHERPA_ONNX_VERSION=$(grep "SHERPA_ONNX_VERSION" ./CMakeLists.txt  | cut -d " " -f 2  | cut -d '"' -f 2)
echo "SHERPA_ONNX_VERSION: ${SHERPA_ONNX_VERSION}"
dst=v${SHERPA_ONNX_VERSION}

if [ -d $dst ]; then
  echo "$dst exists - skipping"
  exit 0
fi

./build-android-arm64-v8a.sh
./build-android-armv7-eabi.sh
./build-android-x86-64.sh
./build-android-x86.sh
./build-ios.sh

mkdir -p $dst/jniLibs/arm64-v8a
cp -v ./build-android-arm64-v8a/install/lib/*.so $dst/jniLibs/arm64-v8a/

mkdir -p $dst/jniLibs/armeabi-v7a
cp -v ./build-android-armv7-eabi/install/lib/*.so $dst/jniLibs/armeabi-v7a/

mkdir -p $dst/jniLibs/x86_64
cp -v ./build-android-x86-64/install/lib/*.so $dst/jniLibs/x86_64

mkdir -p $dst/jniLibs/x86
cp -v ./build-android-x86/install/lib/*.so $dst/jniLibs/x86

mkdir -p $dst/build-ios/
cp -av ./build-ios/sherpa-onnx.xcframework $dst/build-ios/

mkdir -p $dst/build-ios/ios-onnxruntime
cp -av ./build-ios/ios-onnxruntime/onnxruntime.xcframework $dst/build-ios/ios-onnxruntime/

cd $dst

tar cjvf sherpa-onnx-v${SHERPA_ONNX_VERSION}-android.tar.bz2 ./jniLibs

tar cjvf sherpa-onnx-v${SHERPA_ONNX_VERSION}-ios.tar.bz2 ./build-ios
