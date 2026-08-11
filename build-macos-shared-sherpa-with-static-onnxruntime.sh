#!/usr/bin/env bash
#
# This script builds a shared xcframework for macOS with onnxruntime
# statically linked in. The resulting libsherpa-onnx-c-api.dylib is
# self-contained and does NOT require a separate libonnxruntime.dylib.
#
# Differences from build-macos-shared.sh:
#   - build-macos-shared.sh produces a shared libsherpa-onnx-c-api.dylib that
#     DEPENDS on a separate libonnxruntime.dylib (downloaded via cmake at build time).
#     Users must ship both dylibs together.
#   - This script downloads the STATIC onnxruntime library (libonnxruntime.a) and
#     links it into libsherpa-onnx-c-api.dylib. The output is a single self-contained
#     dylib with no external onnxruntime dependency.
#
# When to use which:
#   - build-macos-shared.sh: when you want a smaller sherpa-onnx dylib and are OK
#     shipping onnxruntime separately (e.g., SPM with separate onnxruntime xcframework).
#   - This script: when you want a single dylib with everything included (e.g., for
#     Flutter pub.dev where fewer files and smaller total size matters).

set -ex

dir=build-macos-shared-sherpa-with-static-onnxruntime
mkdir -p $dir
cd $dir

onnxruntime_version=${SHERPA_ONNX_ONNXRUNTIME_VERSION:-1.27.1}
onnxruntime_dir=onnxruntime-static/$onnxruntime_version

if [ ! -f $onnxruntime_dir/lib/libonnxruntime.a ]; then
  mkdir -p $onnxruntime_dir
  pushd $onnxruntime_dir
  wget -c -q https://github.com/csukuangfj/onnxruntime-libs/releases/download/v${onnxruntime_version}/onnxruntime-osx-universal2-static_lib-${onnxruntime_version}.zip
  unzip onnxruntime-osx-universal2-static_lib-${onnxruntime_version}.zip
  mv onnxruntime-osx-universal2-static_lib-${onnxruntime_version}/* .
  rm -rf onnxruntime-osx-universal2-static_lib-${onnxruntime_version}
  rm onnxruntime-osx-universal2-static_lib-${onnxruntime_version}.zip
  popd
fi

export SHERPA_ONNXRUNTIME_LIB_DIR=$PWD/$onnxruntime_dir/lib
export SHERPA_ONNXRUNTIME_INCLUDE_DIR=$PWD/$onnxruntime_dir/include

echo "SHERPA_ONNXRUNTIME_LIB_DIR: $SHERPA_ONNXRUNTIME_LIB_DIR"
echo "SHERPA_ONNXRUNTIME_INCLUDE_DIR: $SHERPA_ONNXRUNTIME_INCLUDE_DIR"

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
  -DCMAKE_CXX_FLAGS="-DSHERPA_ONNX_DISABLE_COREML" \
  -DCMAKE_C_FLAGS="-DSHERPA_ONNX_DISABLE_COREML" \
  -DCMAKE_SHARED_LINKER_FLAGS="-framework Foundation" \
  ../

make -j4
make install
rm -fv ./install/include/cargs.h

echo "Verifying onnxruntime is NOT a dynamic dependency:"
otool -L ./install/lib/libsherpa-onnx-c-api.dylib
if otool -L ./install/lib/libsherpa-onnx-c-api.dylib | grep -q libonnxruntime; then
  echo "ERROR: libonnxruntime is still a dynamic dependency!"
  exit 1
fi
echo "OK: onnxruntime is statically linked"

# Create a framework bundle (like onnxruntime does) so SPM can resolve the module
FRAMEWORK_DIR=SherpaOnnxC.framework
rm -rf $FRAMEWORK_DIR

mkdir -p $FRAMEWORK_DIR/Versions/A/Headers/sherpa-onnx/c-api
mkdir -p $FRAMEWORK_DIR/Versions/A/Modules
mkdir -p $FRAMEWORK_DIR/Versions/A/Resources

# Binary
cp install/lib/libsherpa-onnx-c-api.dylib $FRAMEWORK_DIR/Versions/A/SherpaOnnxC

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
  <string>20260810</string>
  <key>CFBundleShortVersionString</key>
  <string>1.13.5</string>
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

# Fix dylib install name to use framework-relative path
install_name_tool -id @rpath/SherpaOnnxC.framework/Versions/A/SherpaOnnxC $FRAMEWORK_DIR/Versions/A/SherpaOnnxC

rm -rf sherpa-onnx.xcframework

xcodebuild -create-xcframework \
  -framework $FRAMEWORK_DIR \
  -output sherpa-onnx.xcframework

SHERPA_ONNX_VERSION=v$(grep "SHERPA_ONNX_VERSION" ../CMakeLists.txt | cut -d " " -f 2 | cut -d '"' -f 2)

rm -f sherpa-onnx-${SHERPA_ONNX_VERSION}-macos-shared-onnxruntime-static.xcframework.zip
zip -r -y sherpa-onnx-${SHERPA_ONNX_VERSION}-macos-shared-onnxruntime-static.xcframework.zip sherpa-onnx.xcframework

echo "Checksum:"
swift package compute-checksum sherpa-onnx-${SHERPA_ONNX_VERSION}-macos-shared-onnxruntime-static.xcframework.zip | tee checksum.txt
