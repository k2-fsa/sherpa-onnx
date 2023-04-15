#!/usr/bin/env bash
#
# Copyright (c)  2023  Xiaomi Corporation
#
# Please see the end of this file for what files it will generate

SHERPA_ONNX_VERSION=$(grep "SHERPA_ONNX_VERSION" ./CMakeLists.txt  | cut -d " " -f 2  | cut -d '"' -f 2)
echo "SHERPA_ONNX_VERSION: ${SHERPA_ONNX_VERSION}"
dst=v${SHERPA_ONNX_VERSION}

if [ -d $dst ]; then
  echo "$dst exists - skipping"
  exit 0
fi

./build-android-x86-64.sh
./build-android-armv7-eabi.sh
./build-android-x86-64.sh
./build-ios.sh

mkdir -p $dst/jniLibs/arm64-v8a
cp -v ./build-android-arm64-v8a/install/lib/*.so $dst/jniLibs/arm64-v8a/

mkdir -p $dst/jniLibs/armeabi-v7a
cp -v ./build-android-armv7-eabi/install/lib/*.so $dst/jniLibs/armeabi-v7a/

mkdir -p $dst/jniLibs/x86_64
cp -v ./build-android-x86-64/install/lib/*.so $dst/jniLibs/x86_64

mkdir -p $dst/build-ios/
cp -av ./build-ios/sherpa-onnx.xcframework $dst/build-ios/

mkdir -p $dst/build-ios/ios-onnxruntime
cp -av ./build-ios/ios-onnxruntime/onnxruntime.xcframework $dst/build-ios/ios-onnxruntime/

cd $dst

tar cjvf sherpa-onnx-v${SHERPA_ONNX_VERSION}-pre-compiled-android-libs.tar.bz2 ./jniLibs

tar cjvf sherpa-onnx-v${SHERPA_ONNX_VERSION}-pre-compiled-ios-libs.tar.bz2 ./build-ios

# .
# ├── build-ios
# │   ├── ios-onnxruntime
# │   │   └── onnxruntime.xcframework
# │   │       ├── Headers
# │   │       │   ├── cpu_provider_factory.h
# │   │       │   ├── onnxruntime_c_api.h
# │   │       │   ├── onnxruntime_cxx_api.h
# │   │       │   └── onnxruntime_cxx_inline.h
# │   │       ├── Info.plist
# │   │       ├── ios-arm64
# │   │       │   ├── libonnxruntime.a -> onnxruntime.a
# │   │       │   └── onnxruntime.a
# │   │       └── ios-arm64_x86_64-simulator
# │   │           ├── libonnxruntime.a -> onnxruntime.a
# │   │           └── onnxruntime.a
# │   └── sherpa-onnx.xcframework
# │       ├── Headers
# │       │   └── sherpa-onnx
# │       │       └── c-api
# │       │           └── c-api.h
# │       ├── Info.plist
# │       ├── ios-arm64
# │       │   ├── libsherpa-onnx.a -> sherpa-onnx.a
# │       │   └── sherpa-onnx.a
# │       └── ios-arm64_x86_64-simulator
# │           ├── libsherpa-onnx.a -> sherpa-onnx.a
# │           └── sherpa-onnx.a
# ├── jniLibs
# │   ├── arm64-v8a
# │   │   ├── libkaldi-native-fbank-core.so
# │   │   ├── libonnxruntime.so
# │   │   ├── libsherpa-onnx-c-api.so
# │   │   ├── libsherpa-onnx-core.so
# │   │   └── libsherpa-onnx-jni.so
# │   ├── armeabi-v7a
# │   │   ├── libkaldi-native-fbank-core.so
# │   │   ├── libonnxruntime.so
# │   │   ├── libsherpa-onnx-c-api.so
# │   │   ├── libsherpa-onnx-core.so
# │   │   └── libsherpa-onnx-jni.so
# │   └── x86_64
# │       ├── libkaldi-native-fbank-core.so
# │       ├── libonnxruntime.so
# │       ├── libsherpa-onnx-core.so
# │       └── libsherpa-onnx-jni.so
# ├── sherpa-onnx-v1.4.1-pre-compiled-android-libs.tar.bz2
# └── sherpa-onnx-v1.4.1-pre-compiled-ios-libs.tar.bz2
#
# 16 directories, 31 files
#
# 1.4.1 is the current version listed in ./CMakeLists.txt
