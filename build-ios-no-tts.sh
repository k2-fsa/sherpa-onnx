#!/usr/bin/env  bash

set -e

dir=build-ios-no-tts
mkdir -p $dir
cd $dir
onnxruntime_version=1.17.1
onnxruntime_dir=ios-onnxruntime/$onnxruntime_version

if [ ! -f $onnxruntime_dir/onnxruntime.xcframework/ios-arm64/onnxruntime.a ]; then
  mkdir -p $onnxruntime_dir
  pushd $onnxruntime_dir
  wget -c https://github.com/csukuangfj/onnxruntime-libs/releases/download/v${onnxruntime_version}/onnxruntime.xcframework-${onnxruntime_version}.tar.bz2
  tar xvf onnxruntime.xcframework-${onnxruntime_version}.tar.bz2
  rm onnxruntime.xcframework-${onnxruntime_version}.tar.bz2
  cd ..
  ln -sf $onnxruntime_version/onnxruntime.xcframework .
  popd
fi

# First, for simulator
echo "Building for simulator (x86_64)"

export SHERPA_ONNXRUNTIME_LIB_DIR=$PWD/ios-onnxruntime/onnxruntime.xcframework/ios-arm64_x86_64-simulator
export SHERPA_ONNXRUNTIME_INCLUDE_DIR=$PWD/ios-onnxruntime/onnxruntime.xcframework/Headers

echo "SHERPA_ONNXRUNTIME_LIB_DIR: $SHERPA_ONNXRUNTIME_LIB_DIR"
echo "SHERPA_ONNXRUNTIME_INCLUDE_DIR $SHERPA_ONNXRUNTIME_INCLUDE_DIR"

# Note: We use -DENABLE_ARC=1 here to fix the linking error:
#
# The symbol _NSLog is not defined
#

cmake \
  -DSHERPA_ONNX_ENABLE_TTS=OFF \
  -DBUILD_PIPER_PHONMIZE_EXE=OFF \
  -DBUILD_PIPER_PHONMIZE_TESTS=OFF \
  -DBUILD_ESPEAK_NG_EXE=OFF \
  -DBUILD_ESPEAK_NG_TESTS=OFF \
  -S .. \
  -DCMAKE_TOOLCHAIN_FILE=./toolchains/ios.toolchain.cmake \
  -DPLATFORM=SIMULATOR64 \
  -DENABLE_BITCODE=0 \
  -DENABLE_ARC=1 \
  -DENABLE_VISIBILITY=0 \
  -DCMAKE_BUILD_TYPE=Release \
  -DBUILD_SHARED_LIBS=OFF \
  -DSHERPA_ONNX_ENABLE_PYTHON=OFF \
  -DSHERPA_ONNX_ENABLE_TESTS=OFF \
  -DSHERPA_ONNX_ENABLE_CHECK=OFF \
  -DSHERPA_ONNX_ENABLE_PORTAUDIO=OFF \
  -DSHERPA_ONNX_ENABLE_JNI=OFF \
  -DSHERPA_ONNX_ENABLE_C_API=ON \
  -DSHERPA_ONNX_ENABLE_WEBSOCKET=OFF \
  -DDEPLOYMENT_TARGET=13.0 \
  -B build/simulator_x86_64

cmake --build build/simulator_x86_64 -j 4 --verbose

echo "Building for simulator (arm64)"

cmake \
  -DSHERPA_ONNX_ENABLE_TTS=OFF \
  -DBUILD_PIPER_PHONMIZE_EXE=OFF \
  -DBUILD_PIPER_PHONMIZE_TESTS=OFF \
  -DBUILD_ESPEAK_NG_EXE=OFF \
  -DBUILD_ESPEAK_NG_TESTS=OFF \
  -S .. \
  -DCMAKE_TOOLCHAIN_FILE=./toolchains/ios.toolchain.cmake \
  -DPLATFORM=SIMULATORARM64 \
  -DENABLE_BITCODE=0 \
  -DENABLE_ARC=1 \
  -DENABLE_VISIBILITY=0 \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX=./install \
  -DBUILD_SHARED_LIBS=OFF \
  -DSHERPA_ONNX_ENABLE_PYTHON=OFF \
  -DSHERPA_ONNX_ENABLE_TESTS=OFF \
  -DSHERPA_ONNX_ENABLE_CHECK=OFF \
  -DSHERPA_ONNX_ENABLE_PORTAUDIO=OFF \
  -DSHERPA_ONNX_ENABLE_JNI=OFF \
  -DSHERPA_ONNX_ENABLE_C_API=ON \
  -DSHERPA_ONNX_ENABLE_WEBSOCKET=OFF \
  -DDEPLOYMENT_TARGET=13.0 \
  -B build/simulator_arm64

cmake --build build/simulator_arm64 -j 4 --verbose

echo "Building for arm64"

export SHERPA_ONNXRUNTIME_LIB_DIR=$PWD/ios-onnxruntime/onnxruntime.xcframework/ios-arm64


cmake \
  -DSHERPA_ONNX_ENABLE_TTS=OFF \
  -DBUILD_PIPER_PHONMIZE_EXE=OFF \
  -DBUILD_PIPER_PHONMIZE_TESTS=OFF \
  -DBUILD_ESPEAK_NG_EXE=OFF \
  -DBUILD_ESPEAK_NG_TESTS=OFF \
  -S .. \
  -DCMAKE_TOOLCHAIN_FILE=./toolchains/ios.toolchain.cmake \
  -DPLATFORM=OS64 \
  -DENABLE_BITCODE=0 \
  -DENABLE_ARC=1 \
  -DENABLE_VISIBILITY=0 \
  -DCMAKE_INSTALL_PREFIX=./install \
  -DCMAKE_BUILD_TYPE=Release \
  -DBUILD_SHARED_LIBS=OFF \
  -DSHERPA_ONNX_ENABLE_PYTHON=OFF \
  -DSHERPA_ONNX_ENABLE_TESTS=OFF \
  -DSHERPA_ONNX_ENABLE_CHECK=OFF \
  -DSHERPA_ONNX_ENABLE_PORTAUDIO=OFF \
  -DSHERPA_ONNX_ENABLE_JNI=OFF \
  -DSHERPA_ONNX_ENABLE_C_API=ON \
  -DSHERPA_ONNX_ENABLE_WEBSOCKET=OFF \
  -DDEPLOYMENT_TARGET=13.0 \
  -B build/os64

cmake --build build/os64 -j 4
# Generate headers for sherpa-onnx.xcframework
cmake --build build/os64 --target install

echo "Generate xcframework"

mkdir -p "build/simulator/lib"
for f in libkaldi-native-fbank-core.a libsherpa-onnx-c-api.a libsherpa-onnx-core.a \
         libsherpa-onnx-fst.a libsherpa-onnx-kaldifst-core.a libkaldi-decoder-core.a; do
  lipo -create build/simulator_arm64/lib/${f} \
               build/simulator_x86_64/lib/${f} \
       -output build/simulator/lib/${f}
done

# Merge archive first, because the following xcodebuild create xcframework
# cannot accept multi archive with the same architecture.
libtool -static -o build/simulator/sherpa-onnx.a \
  build/simulator/lib/libkaldi-native-fbank-core.a \
  build/simulator/lib/libsherpa-onnx-c-api.a \
  build/simulator/lib/libsherpa-onnx-core.a  \
  build/simulator/lib/libsherpa-onnx-fst.a   \
  build/simulator/lib/libsherpa-onnx-kaldifst-core.a \
  build/simulator/lib/libkaldi-decoder-core.a

libtool -static -o build/os64/sherpa-onnx.a \
  build/os64/lib/libkaldi-native-fbank-core.a \
  build/os64/lib/libsherpa-onnx-c-api.a \
  build/os64/lib/libsherpa-onnx-core.a \
  build/os64/lib/libsherpa-onnx-fst.a   \
  build/os64/lib/libsherpa-onnx-kaldifst-core.a \
  build/os64/lib/libkaldi-decoder-core.a

rm -rf sherpa-onnx.xcframework

xcodebuild -create-xcframework \
      -library "build/os64/sherpa-onnx.a" \
      -library "build/simulator/sherpa-onnx.a" \
      -output sherpa-onnx.xcframework

# Copy Headers
mkdir -p sherpa-onnx.xcframework/Headers
cp -av install/include/* sherpa-onnx.xcframework/Headers

pushd sherpa-onnx.xcframework/ios-arm64_x86_64-simulator
ln -s sherpa-onnx.a libsherpa-onnx.a
popd

pushd sherpa-onnx.xcframework/ios-arm64
ln -s sherpa-onnx.a libsherpa-onnx.a
