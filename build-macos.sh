#!/usr/bin/env  bash

set -ex

dir=build-macos
mkdir -p $dir
cd $dir

cmake \
  -DSHERPA_ONNX_ENABLE_BINARY=OFF \
  -DSHERPA_ONNX_BUILD_C_API_EXAMPLES=OFF \
  -DCMAKE_OSX_ARCHITECTURES="arm64;x86_64" \
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
  ../

make -j4
make install
rm -fv ./install/include/cargs.h

libtool -static -o ./install/lib/libsherpa-onnx.a \
  ./install/lib/libsherpa-onnx-c-api.a \
  ./install/lib/libsherpa-onnx-core.a \
  ./install/lib/libkaldi-native-fbank-core.a \
  ./install/lib/libkissfft-float.a \
  ./install/lib/libsherpa-onnx-fstfar.a \
  ./install/lib/libsherpa-onnx-fst.a \
  ./install/lib/libsherpa-onnx-kaldifst-core.a \
  ./install/lib/libkaldi-decoder-core.a \
  ./install/lib/libucd.a \
  ./install/lib/libpiper_phonemize.a \
  ./install/lib/libespeak-ng.a \
  ./install/lib/libssentencepiece_core.a

# Create framework directory structure
rm -rf sherpa-onnx.framework
mkdir -p sherpa-onnx.framework/Headers/sherpa-onnx/c-api
mkdir -p sherpa-onnx.framework/Modules

# Copy binary (rename from libsherpa-onnx.a to sherpa-onnx)
cp -v install/lib/libsherpa-onnx.a sherpa-onnx.framework/sherpa-onnx

# Copy headers into multi-level path
cp -v install/include/sherpa-onnx/c-api/c-api.h sherpa-onnx.framework/Headers/sherpa-onnx/c-api/

# Create module map
cat > sherpa-onnx.framework/Modules/module.modulemap << 'EOF'
module SherpaOnnxC {
    header "sherpa-onnx/c-api/c-api.h"
    export *
}
EOF

rm -rf sherpa-onnx.xcframework
xcodebuild -create-xcframework \
  -framework sherpa-onnx.framework \
  -output sherpa-onnx.xcframework

# Remove the module map from the install directory to prevent swiftc from
# auto-discovering it. The module map is only needed inside the xcframework
# (used by SPM). For direct swiftc builds, the bridging header is used instead.
rm -fv ./install/include/sherpa-onnx/c-api/module.modulemap

SHERPA_ONNX_VERSION=v$(grep "SHERPA_ONNX_VERSION" ../CMakeLists.txt | cut -d " " -f 2 | cut -d '"' -f 2)

rm -f sherpa-onnx-${SHERPA_ONNX_VERSION}-macos-static.xcframework.zip
zip -r -y sherpa-onnx-${SHERPA_ONNX_VERSION}-macos-static.xcframework.zip sherpa-onnx.xcframework

echo "Checksum:"
swift package compute-checksum sherpa-onnx-${SHERPA_ONNX_VERSION}-macos-static.xcframework.zip | tee checksum.txt
