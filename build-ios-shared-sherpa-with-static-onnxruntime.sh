#!/usr/bin/env bash
#
# This script builds a shared xcframework for iOS with onnxruntime
# statically linked in. The resulting libsherpa-onnx-c-api.dylib is
# self-contained and does NOT require a separate onnxruntime framework.
#
# Differences from build-ios-shared.sh:
#   - build-ios-shared.sh produces a shared libsherpa-onnx-c-api.dylib that
#     DEPENDS on a separate onnxruntime.framework (shared, downloaded as xcframework).
#     Users must ship both frameworks together.
#   - This script downloads the STATIC onnxruntime iOS xcframework and links it into
#     libsherpa-onnx-c-api.dylib. The output is a single self-contained dylib with
#     no external onnxruntime dependency.
#   - This script also strips debug symbols (strip -x) to reduce binary size, keeping
#     the total xcframework under 100MB for pub.dev publishing.
#
# When to use which:
#   - build-ios-shared.sh: when you want a smaller sherpa-onnx dylib and are OK
#     shipping onnxruntime separately (e.g., SPM with separate onnxruntime xcframework).
#   - This script: when you want a single dylib with everything included (e.g., for
#     Flutter pub.dev where fewer files and size limits matter).

set -e

dir=build-ios-shared-sherpa-with-static-onnxruntime
mkdir -p $dir
cd $dir
onnxruntime_version=${SHERPA_ONNX_ONNXRUNTIME_VERSION:-1.27.1}
onnxruntime_dir=ios-onnxruntime/$onnxruntime_version

CMAKE_VERBOSE_MAKEFILE=${CMAKE_VERBOSE_MAKEFILE:-OFF}

if [ ! -f $onnxruntime_dir/onnxruntime.xcframework/ios-arm64/onnxruntime.framework/onnxruntime ]; then
  mkdir -p $onnxruntime_dir
  pushd $onnxruntime_dir
  wget -c https://github.com/csukuangfj/onnxruntime-libs/releases/download/v${onnxruntime_version}/onnxruntime-ios-static-xcframework-${onnxruntime_version}.xcframework.zip
  unzip onnxruntime-ios-static-xcframework-${onnxruntime_version}.xcframework.zip
  rm onnxruntime-ios-static-xcframework-${onnxruntime_version}.xcframework.zip
  cd ..
  ln -sf $onnxruntime_version/onnxruntime.xcframework .
  popd
fi

# First, for simulator (x86_64)
echo "Building for simulator (x86_64)"

export SHERPA_ONNXRUNTIME_LIB_DIR=$PWD/ios-onnxruntime/onnxruntime.xcframework/ios-arm64_x86_64-simulator
export SHERPA_ONNXRUNTIME_INCLUDE_DIR=$PWD/ios-onnxruntime/onnxruntime.xcframework/ios-arm64_x86_64-simulator/onnxruntime.framework/Headers

echo "SHERPA_ONNXRUNTIME_LIB_DIR: $SHERPA_ONNXRUNTIME_LIB_DIR"
echo "SHERPA_ONNXRUNTIME_INCLUDE_DIR: $SHERPA_ONNXRUNTIME_INCLUDE_DIR"

echo "Building for simulator (x86_64)"

if [[ ! -f build/simulator_x86_64/install/lib/libsherpa-onnx-c-api.dylib ]]; then
  cmake \
    -DSHERPA_ONNX_ENABLE_BINARY=OFF \
    -DBUILD_PIPER_PHONMIZE_EXE=OFF \
    -DBUILD_PIPER_PHONMIZE_TESTS=OFF \
    -DBUILD_ESPEAK_NG_EXE=OFF \
    -DBUILD_ESPEAK_NG_TESTS=OFF \
    -S .. -D CMAKE_VERBOSE_MAKEFILE=$CMAKE_VERBOSE_MAKEFILE \
    -DCMAKE_TOOLCHAIN_FILE=./toolchains/ios.toolchain.cmake \
    -DPLATFORM=SIMULATOR64 \
    -DENABLE_BITCODE=0 \
    -DENABLE_ARC=1 \
    -DENABLE_VISIBILITY=1 \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX=./build/simulator_x86_64/install \
    -DBUILD_SHARED_LIBS=ON \
    -DSHERPA_ONNX_ENABLE_PYTHON=OFF \
    -DSHERPA_ONNX_ENABLE_TESTS=OFF \
    -DSHERPA_ONNX_ENABLE_CHECK=OFF \
    -DSHERPA_ONNX_ENABLE_PORTAUDIO=OFF \
    -DSHERPA_ONNX_ENABLE_JNI=OFF \
    -DSHERPA_ONNX_ENABLE_C_API=ON \
    -DSHERPA_ONNX_ENABLE_WEBSOCKET=OFF \
    -DDEPLOYMENT_TARGET=13.0 \
    -B build/simulator_x86_64

  cmake --build build/simulator_x86_64 -j 4 --target install
else
  echo "Skip building for simulator (x86_64)"
fi

echo "Building for simulator (arm64)"

if [[ ! -f build/simulator_arm64/install/lib/libsherpa-onnx-c-api.dylib ]]; then
  cmake \
    -DSHERPA_ONNX_ENABLE_BINARY=OFF \
    -DBUILD_PIPER_PHONMIZE_EXE=OFF \
    -DBUILD_PIPER_PHONMIZE_TESTS=OFF \
    -DBUILD_ESPEAK_NG_EXE=OFF \
    -DBUILD_ESPEAK_NG_TESTS=OFF \
    -S .. -D CMAKE_VERBOSE_MAKEFILE=$CMAKE_VERBOSE_MAKEFILE \
    -DCMAKE_TOOLCHAIN_FILE=./toolchains/ios.toolchain.cmake \
    -DPLATFORM=SIMULATORARM64 \
    -DENABLE_BITCODE=0 \
    -DENABLE_ARC=1 \
    -DENABLE_VISIBILITY=1 \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX=./build/simulator_arm64/install \
    -DBUILD_SHARED_LIBS=ON \
    -DSHERPA_ONNX_ENABLE_PYTHON=OFF \
    -DSHERPA_ONNX_ENABLE_TESTS=OFF \
    -DSHERPA_ONNX_ENABLE_CHECK=OFF \
    -DSHERPA_ONNX_ENABLE_PORTAUDIO=OFF \
    -DSHERPA_ONNX_ENABLE_JNI=OFF \
    -DSHERPA_ONNX_ENABLE_C_API=ON \
    -DSHERPA_ONNX_ENABLE_WEBSOCKET=OFF \
    -DDEPLOYMENT_TARGET=13.0 \
    -B build/simulator_arm64

  cmake --build build/simulator_arm64 -j 4 --target install
else
  echo "Skip building for simulator (arm64)"
fi

echo "Building for arm64"

if [[ ! -f build/os64/install/lib/libsherpa-onnx-c-api.dylib ]]; then
  export SHERPA_ONNXRUNTIME_LIB_DIR=$PWD/ios-onnxruntime/onnxruntime.xcframework/ios-arm64
  export SHERPA_ONNXRUNTIME_INCLUDE_DIR=$PWD/ios-onnxruntime/onnxruntime.xcframework/ios-arm64/onnxruntime.framework/Headers

  cmake \
    -DSHERPA_ONNX_ENABLE_BINARY=OFF \
    -DBUILD_PIPER_PHONMIZE_EXE=OFF \
    -DBUILD_PIPER_PHONMIZE_TESTS=OFF \
    -DBUILD_ESPEAK_NG_EXE=OFF \
    -DBUILD_ESPEAK_NG_TESTS=OFF \
    -S .. -D CMAKE_VERBOSE_MAKEFILE=$CMAKE_VERBOSE_MAKEFILE \
    -DCMAKE_TOOLCHAIN_FILE=./toolchains/ios.toolchain.cmake \
    -DPLATFORM=OS64 \
    -DENABLE_BITCODE=0 \
    -DENABLE_ARC=1 \
    -DENABLE_VISIBILITY=1 \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX=./build/os64/install \
    -DBUILD_SHARED_LIBS=ON \
    -DSHERPA_ONNX_ENABLE_PYTHON=OFF \
    -DSHERPA_ONNX_ENABLE_TESTS=OFF \
    -DSHERPA_ONNX_ENABLE_CHECK=OFF \
    -DSHERPA_ONNX_ENABLE_PORTAUDIO=OFF \
    -DSHERPA_ONNX_ENABLE_JNI=OFF \
    -DSHERPA_ONNX_ENABLE_C_API=ON \
    -DSHERPA_ONNX_ENABLE_WEBSOCKET=OFF \
    -DDEPLOYMENT_TARGET=13.0 \
    -B build/os64

  cmake --build build/os64 -j 4 --target install
else
  echo "Skip building for arm64"
fi

echo "Collect dynamic libraries"
mkdir -p ios-arm64 ios-arm64-simulator ios-x86_64-simulator

cp -v ./build/os64/install/lib/libsherpa-onnx-c-api.dylib ios-arm64/
cp -v ./build/simulator_arm64/install/lib/libsherpa-onnx-c-api.dylib ios-arm64-simulator/
cp -v ./build/simulator_x86_64/install/lib/libsherpa-onnx-c-api.dylib ios-x86_64-simulator/

# Strip debug symbols to reduce size
strip -x ios-arm64/libsherpa-onnx-c-api.dylib
strip -x ios-arm64-simulator/libsherpa-onnx-c-api.dylib
strip -x ios-x86_64-simulator/libsherpa-onnx-c-api.dylib

# Create fat simulator binary
rm -rf ios-arm64_x86_64-simulator
mkdir ios-arm64_x86_64-simulator

lipo \
  -create \
    ios-arm64-simulator/libsherpa-onnx-c-api.dylib \
    ios-x86_64-simulator/libsherpa-onnx-c-api.dylib \
  -output \
    ios-arm64_x86_64-simulator/libsherpa-onnx-c-api.dylib

rm -rf SherpaOnnxC.xcframework

# Create framework bundles so SPM can resolve the module
create_framework() {
  local lib_path=$1
  local output_dir=$2

  local fw_dir=$output_dir/SherpaOnnxC.framework
  rm -rf $fw_dir

  mkdir -p $fw_dir/Headers/sherpa-onnx/c-api
  mkdir -p $fw_dir/Modules

  cp $lib_path $fw_dir/SherpaOnnxC
  cp build/os64/install/include/sherpa-onnx/c-api/c-api.h $fw_dir/Headers/sherpa-onnx/c-api/

  cat > $fw_dir/Modules/module.modulemap << 'MEOF'
framework module SherpaOnnxC {
  header "sherpa-onnx/c-api/c-api.h"
  export *
}
MEOF

  cat > $fw_dir/Info.plist << 'PEOF'
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
  <string>20260818</string>
  <key>CFBundleShortVersionString</key>
  <string>1.13.6</string>
  <key>MinimumOSVersion</key>
  <string>13.0</string>
  <key>CFBundleSupportedPlatforms</key>
  <array>
    <string>iPhoneOS</string>
  </array>
</dict>
</plist>
PEOF

  # Fix dylib install name
  install_name_tool -id @rpath/SherpaOnnxC.framework/SherpaOnnxC $fw_dir/SherpaOnnxC

  # Ad-hoc sign the framework binary so Xcode can embed and re-sign it
  codesign --force --sign - $fw_dir/SherpaOnnxC
}

create_framework ios-arm64/libsherpa-onnx-c-api.dylib ios-arm64
create_framework ios-arm64_x86_64-simulator/libsherpa-onnx-c-api.dylib ios-arm64_x86_64-simulator

xcodebuild -create-xcframework \
  -framework "ios-arm64/SherpaOnnxC.framework" \
  -framework "ios-arm64_x86_64-simulator/SherpaOnnxC.framework" \
  -output SherpaOnnxC.xcframework

cd SherpaOnnxC.xcframework
echo "PWD: $PWD"
ls -lh
echo "---"
ls -lh */*

cd ..

SHERPA_ONNX_VERSION=v$(grep "SHERPA_ONNX_VERSION" ../CMakeLists.txt | cut -d " " -f 2 | cut -d '"' -f 2)
rm -f sherpa-onnx-${SHERPA_ONNX_VERSION}-ios-shared-onnxruntime-static.xcframework.zip
zip -r -y sherpa-onnx-${SHERPA_ONNX_VERSION}-ios-shared-onnxruntime-static.xcframework.zip SherpaOnnxC.xcframework

echo "Checksum:"
swift package compute-checksum sherpa-onnx-${SHERPA_ONNX_VERSION}-ios-shared-onnxruntime-static.xcframework.zip | tee checksum.txt
