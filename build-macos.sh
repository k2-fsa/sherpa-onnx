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

# Rename to match the expected library name for linking with -l sherpa-onnx-c-api
mv -v install/lib/libsherpa-onnx.a install/lib/libsherpa-onnx-c-api.a

# Create a framework bundle (like onnxruntime does) so SPM can resolve the module
FRAMEWORK_DIR=SherpaOnnxC.framework
rm -rf $FRAMEWORK_DIR

mkdir -p $FRAMEWORK_DIR/Versions/A/Headers/sherpa-onnx/c-api
mkdir -p $FRAMEWORK_DIR/Versions/A/Modules
mkdir -p $FRAMEWORK_DIR/Versions/A/Resources

# Binary
cp install/lib/libsherpa-onnx-c-api.a $FRAMEWORK_DIR/Versions/A/SherpaOnnxC

# Headers (preserve nested path for #include "sherpa-onnx/c-api/c-api.h")
cp install/include/sherpa-onnx/c-api/c-api.h $FRAMEWORK_DIR/Versions/A/Headers/sherpa-onnx/c-api/

# Modulemap
cat > $FRAMEWORK_DIR/Versions/A/Modules/module.modulemap << 'EOF'
framework module SherpaOnnxC {
  header "sherpa-onnx/c-api/c-api.h"
  export *
}
EOF

# Info.plist
cat > $FRAMEWORK_DIR/Versions/A/Resources/Info.plist << 'EOF'
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>CFBundleIdentifier</key>
  <string>com.k2-fsa.sherpa-onnx</string>
  <key>CFBundleName</key>
  <string>SherpaOnnxC</string>
  <key>CFBundlePackageType</key>
  <string>FMWK</string>
  <key>CFBundleExecutable</key>
  <string>SherpaOnnxC</string>
  <key>CFBundleVersion</key>
  <string>20260901</string>
  <key>CFBundleShortVersionString</key>
  <string>1.13.7</string>
</dict>
</plist>
EOF

# Versioned symlinks
pushd $FRAMEWORK_DIR/Versions
ln -sf A Current
popd

ln -sf Versions/Current/SherpaOnnxC $FRAMEWORK_DIR/SherpaOnnxC
ln -sf Versions/Current/Headers $FRAMEWORK_DIR/Headers
ln -sf Versions/Current/Modules $FRAMEWORK_DIR/Modules
ln -sf Versions/Current/Resources $FRAMEWORK_DIR/Resources

rm -rf sherpa-onnx.xcframework

xcodebuild -create-xcframework \
  -framework $FRAMEWORK_DIR \
  -output sherpa-onnx.xcframework

SHERPA_ONNX_VERSION=v$(grep "SHERPA_ONNX_VERSION" ../CMakeLists.txt | cut -d " " -f 2 | cut -d '"' -f 2)

rm -f sherpa-onnx-${SHERPA_ONNX_VERSION}-macos-static.xcframework.zip
zip -r -y sherpa-onnx-${SHERPA_ONNX_VERSION}-macos-static.xcframework.zip sherpa-onnx.xcframework

echo "Checksum:"
swift package compute-checksum sherpa-onnx-${SHERPA_ONNX_VERSION}-macos-static.xcframework.zip | tee checksum.txt
