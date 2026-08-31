#!/usr/bin/env bash
#
# This script builds a shared xcframework for macOS with onnxruntime
# statically linked in. The resulting libsherpa-onnx-c-api.dylib is
# self-contained and does NOT require a separate libonnxruntime.dylib.

set -ex

dir=build-macos-shared-sherpa-with-static-onnxruntime
mkdir -p $dir
cd $dir

onnxruntime_version=${SHERPA_ONNX_ONNXRUNTIME_VERSION:-1.28.1}
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
rm -rf sherpa-onnx.xcframework

# Fix dylib install name
install_name_tool -id @rpath/libsherpa-onnx-c-api.dylib ./install/lib/libsherpa-onnx-c-api.dylib

# Ad-hoc sign the dylib
codesign --force --sign - ./install/lib/libsherpa-onnx-c-api.dylib

# Create modulemap for SPM
cat > ./install/include/module.modulemap << 'EOF'
module SherpaOnnxC {
  header "sherpa-onnx/c-api/c-api.h"
  export *
}
EOF

xcodebuild -create-xcframework \
  -library ./install/lib/libsherpa-onnx-c-api.dylib \
  -headers ./install/include \
  -output sherpa-onnx.xcframework

SHERPA_ONNX_VERSION=v$(grep "SHERPA_ONNX_VERSION" ../CMakeLists.txt | cut -d " " -f 2 | cut -d '"' -f 2)

rm -f sherpa-onnx-${SHERPA_ONNX_VERSION}-macos-shared-onnxruntime-static.xcframework.zip
zip -r -y sherpa-onnx-${SHERPA_ONNX_VERSION}-macos-shared-onnxruntime-static.xcframework.zip sherpa-onnx.xcframework

echo "Checksum:"
swift package compute-checksum sherpa-onnx-${SHERPA_ONNX_VERSION}-macos-shared-onnxruntime-static.xcframework.zip | tee checksum.txt
