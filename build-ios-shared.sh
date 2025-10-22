#!/usr/bin/env  bash
#
# Note: This script is to build sherpa-onnx for flutter/dart, which requires
# us to use shared libraries for sherpa-onnx.
#
# Note: We still use static libraries for onnxruntime.

set -e

dir=build-ios-shared
mkdir -p $dir
cd $dir
onnxruntime_version=1.17.1
onnxruntime_dir=ios-onnxruntime/$onnxruntime_version

SHERPA_ONNX_GITHUB=github.com

if [ "$SHERPA_ONNX_GITHUB_MIRROW" == true ]; then
    SHERPA_ONNX_GITHUB=hub.nuaa.cf
fi

if [ ! -z CMAKE_VERBOSE_MAKEFILE ]; then
  CMAKE_VERBOSE_MAKEFILE=ON
else
  CMAKE_VERBOSE_MAKEFILE=OFF
fi

if [ ! -f $onnxruntime_dir/onnxruntime.xcframework/ios-arm64/onnxruntime.a ]; then
  mkdir -p $onnxruntime_dir
  pushd $onnxruntime_dir
  wget -c https://${SHERPA_ONNX_GITHUB}/csukuangfj/onnxruntime-libs/releases/download/v${onnxruntime_version}/onnxruntime.xcframework-${onnxruntime_version}.tar.bz2
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

echo "Collect dynamic libraries "
mkdir -p ios-arm64 ios-arm64-simulator ios-x86_64-simulator

cp -v ./build/os64/install/lib/libsherpa-onnx-c-api.dylib ios-arm64/
cp -v ./build/simulator_arm64/install/lib/libsherpa-onnx-c-api.dylib ios-arm64-simulator/
cp -v .//build/simulator_x86_64/install/lib/libsherpa-onnx-c-api.dylib ios-x86_64-simulator/

# see https://github.com/k2-fsa/sherpa-onnx/issues/1172#issuecomment-2439662662
rm -rf ios-arm64_x86_64-simulator
mkdir ios-arm64_x86_64-simulator

lipo \
  -create \
    ios-arm64-simulator/libsherpa-onnx-c-api.dylib \
    ios-x86_64-simulator/libsherpa-onnx-c-api.dylib \
  -output \
    ios-arm64_x86_64-simulator/libsherpa-onnx-c-api.dylib

pushd ios-arm64
rm -rf sherpa_onnx.framework
mkdir sherpa_onnx.framework

lipo \
  -create \
    libsherpa-onnx-c-api.dylib \
  -output \
    sherpa_onnx

mv sherpa_onnx sherpa_onnx.framework/
cd sherpa_onnx.framework

install_name_tool \
  -change @rpath/libsherpa-onnx-c-api.dylib @rpath/sherpa_onnx.framework/sherpa_onnx \
  sherpa_onnx

install_name_tool \
  -id "@rpath/sherpa_onnx.framework/sherpa_onnx" \
  sherpa_onnx

chmod +x sherpa_onnx
popd

pushd ios-arm64_x86_64-simulator
rm -rf sherpa_onnx.framework
mkdir sherpa_onnx.framework

lipo \
  -create \
    libsherpa-onnx-c-api.dylib \
  -output \
    sherpa_onnx

mv sherpa_onnx sherpa_onnx.framework/
cd sherpa_onnx.framework
install_name_tool \
  -change @rpath/libsherpa-onnx-c-api.dylib @rpath/sherpa_onnx.framework/sherpa_onnx \
  sherpa_onnx

install_name_tool \
  -id "@rpath/sherpa_onnx.framework/sherpa_onnx" \
  sherpa_onnx

chmod +x sherpa_onnx
popd

for d in ios-arm64_x86_64-simulator ios-arm64; do
  dst=$d/sherpa_onnx.framework

  # The Info.plist is modified from
  # https://github.com/Spicely/flutter_openim_sdk_ffi/blob/main/ios/openim_sdk_ffi.framework/Info.plist
  cat >$dst/Info.plist <<EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
	<key>CFBundleName</key>
	<string>sherpa_onnx</string>
	<key>DTSDKName</key>
	<string>iphoneos17.0</string>
	<key>DTXcode</key>
	<string>1501</string>
	<key>DTSDKBuild</key>
	<string>21A326</string>
	<key>CFBundleDevelopmentRegion</key>
	<string>en</string>
	<key>CFBundleVersion</key>
	<string>1</string>
	<key>BuildMachineOSBuild</key>
	<string>23B81</string>
	<key>DTPlatformName</key>
	<string>iphoneos</string>
	<key>CFBundlePackageType</key>
	<string>FMWK</string>
	<key>CFBundleShortVersionString</key>
	<string>1.12.15</string>
	<key>CFBundleSupportedPlatforms</key>
	<array>
		<string>iPhoneOS</string>
	</array>
	<key>CFBundleInfoDictionaryVersion</key>
	<string>6.0</string>
	<key>CFBundleExecutable</key>
	<string>sherpa_onnx</string>
	<key>DTCompiler</key>
	<string>com.apple.compilers.llvm.clang.1_0</string>
	<key>UIRequiredDeviceCapabilities</key>
	<array>
		<string>arm64</string>
	</array>
	<key>MinimumOSVersion</key>
	<string>13.0</string>
	<key>CFBundleIdentifier</key>
	<string>com.k2fsa.sherpa.onnx</string>
	<key>UIDeviceFamily</key>
	<array>
		<integer>1</integer>
		<integer>2</integer>
	</array>
	<key>CFBundleSignature</key>
	<string>????</string>
	<key>DTPlatformVersion</key>
	<string>17.0</string>
	<key>DTXcodeBuild</key>
	<string>15A507</string>
	<key>DTPlatformBuild</key>
	<string>21A326</string>
	<key>SupportedArchitectures</key>
	<array>
		<string>arm64</string>
		<string>x86_64</string>
	</array>
	<key>SupportedPlatform</key>
	<string>ios</string>
</dict>
</plist>
EOF
done

rm -rf sherpa_onnx.xcframework
xcodebuild -create-xcframework \
  -framework ios-arm64/sherpa_onnx.framework \
  -framework ios-arm64_x86_64-simulator/sherpa_onnx.framework \
  -output sherpa_onnx.xcframework

cd sherpa_onnx.xcframework
echo "PWD: $PWD"
ls -lh
echo "---"
ls -lh */*
