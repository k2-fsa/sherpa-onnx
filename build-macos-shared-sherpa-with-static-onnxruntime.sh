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

# Create xcframework with bare library (like merman).
# macOS does not use shallow bundles, so we cannot use a flat framework
# structure. Instead, use a bare library without a framework wrapper.
# This avoids the "expected Versions/Current/Resources/Info.plist" error.

rm -rf sherpa-onnx.xcframework

# Fix dylib install name
install_name_tool -id @rpath/libsherpa-onnx-c-api.dylib ./install/lib/libsherpa-onnx-c-api.dylib

# Ad-hoc sign the dylib so Xcode can embed and re-sign it
codesign --force --sign - ./install/lib/libsherpa-onnx-c-api.dylib

xcodebuild -create-xcframework \
  -library ./install/lib/libsherpa-onnx-c-api.dylib \
  -headers ./install/include \
  -output sherpa-onnx.xcframework

SHERPA_ONNX_VERSION=v$(grep "SHERPA_ONNX_VERSION" ../CMakeLists.txt | cut -d " " -f 2 | cut -d '"' -f 2)

rm -f sherpa-onnx-${SHERPA_ONNX_VERSION}-macos-shared-onnxruntime-static.xcframework.zip
zip -r -y sherpa-onnx-${SHERPA_ONNX_VERSION}-macos-shared-onnxruntime-static.xcframework.zip sherpa-onnx.xcframework

echo "Checksum:"
swift package compute-checksum sherpa-onnx-${SHERPA_ONNX_VERSION}-macos-shared-onnxruntime-static.xcframework.zip | tee checksum.txt
