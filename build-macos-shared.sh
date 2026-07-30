#!/usr/bin/env  bash
#
# Build a shared xcframework for macOS (arm64 + x86_64).
# This is used by the Flutter macOS plugin.

set -ex

dir=build-macos-shared
mkdir -p $dir
cd $dir

cmake \
  -DSHERPA_ONNX_ENABLE_BINARY=OFF \
  -DSHERPA_ONNX_BUILD_C_API_EXAMPLES=OFF \
  -DCMAKE_OSX_ARCHITECTURES="arm64;x86_64" \
  -DCMAKE_INSTALL_PREFIX=./install \
  -DCMAKE_BUILD_TYPE=Release \
  -DBUILD_SHARED_LIBS=ON \
  -DSHERPA_ONNX_ENABLE_PYTHON=OFF \
  -DSHERPA_ONNX_ENABLE_TESTS=OFF \
  -DSHERPA_ONNX_ENABLE_CHECK=OFF \
  -DSHERPA_ONNX_ENABLE_PORTAUDIO=OFF \
  -DSHERPA_ONNX_ENABLE_JNI=OFF \
  -DSHERPA_ONNX_ENABLE_C_API=ON \
  -DSHERPA_ONNX_ENABLE_WEBSOCKET=OFF \
  ../

make -j4
make install
rm -fv ./install/include/cargs.h
cp -v ../sherpa-onnx/c-api/module.modulemap ./install/include/sherpa-onnx/c-api/

xcodebuild -create-xcframework \
  -library install/lib/libsherpa-onnx-c-api.dylib \
  -headers install/include/sherpa-onnx/c-api \
  -output sherpa-onnx.xcframework

SHERPA_ONNX_VERSION=v$(grep "SHERPA_ONNX_VERSION" ../CMakeLists.txt | cut -d " " -f 2 | cut -d '"' -f 2)

rm -f sherpa-onnx-${SHERPA_ONNX_VERSION}-macos-shared.xcframework.zip
zip -r -y sherpa-onnx-${SHERPA_ONNX_VERSION}-macos-shared.xcframework.zip sherpa-onnx.xcframework

echo "Checksum:"
swift package compute-checksum sherpa-onnx-${SHERPA_ONNX_VERSION}-macos-shared.xcframework.zip | tee checksum.txt
