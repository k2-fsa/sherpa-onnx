#!/usr/bin/env bash
set -ex

dir=$PWD/build-android-arm64-v8a

mkdir -p $dir
cd $dir

# Note from https://github.com/Tencent/ncnn/wiki/how-to-build#build-for-android
# (optional) remove the hardcoded debug flag in Android NDK android-ndk
# issue: https://github.com/android/ndk/issues/243
#
# open $ANDROID_NDK/build/cmake/android.toolchain.cmake for ndk < r23
# or $ANDROID_NDK/build/cmake/android-legacy.toolchain.cmake for ndk >= r23
#
# delete "-g" line
#
# list(APPEND ANDROID_COMPILER_FLAGS
#   -g
#   -DANDROID

if [ -z $ANDROID_NDK ]; then
  ANDROID_NDK=/star-fj/fangjun/software/android-sdk/ndk/22.1.7171670
  # or use
  # ANDROID_NDK=/star-fj/fangjun/software/android-ndk
  #
  # Inside the $ANDROID_NDK directory, you can find a binary ndk-build
  # and some other files like the file "build/cmake/android.toolchain.cmake"

  if [ ! -d $ANDROID_NDK ]; then
    # For macOS, I have installed Android Studio, select the menu
    # Tools -> SDK manager -> Android SDK
    # and set "Android SDK location" to /Users/fangjun/software/my-android
    ANDROID_NDK=/Users/fangjun/software/my-android/ndk/22.1.7171670
  fi
fi

if [ ! -d $ANDROID_NDK ]; then
  echo Please set the environment variable ANDROID_NDK before you run this script
  exit 1
fi

echo "ANDROID_NDK: $ANDROID_NDK"
sleep 1
onnxruntime_version=1.17.1

if [ ! -f $onnxruntime_version/jni/arm64-v8a/libonnxruntime.so ]; then
  mkdir -p $onnxruntime_version
  pushd $onnxruntime_version
  wget -q https://github.com/csukuangfj/onnxruntime-libs/releases/download/v${onnxruntime_version}/onnxruntime-android-${onnxruntime_version}.zip
  unzip onnxruntime-android-${onnxruntime_version}.zip
  rm onnxruntime-android-${onnxruntime_version}.zip
  popd
fi

export SHERPA_ONNXRUNTIME_LIB_DIR=$dir/$onnxruntime_version/jni/arm64-v8a/
export SHERPA_ONNXRUNTIME_INCLUDE_DIR=$dir/$onnxruntime_version/headers/

echo "SHERPA_ONNXRUNTIME_LIB_DIR: $SHERPA_ONNXRUNTIME_LIB_DIR"
echo "SHERPA_ONNXRUNTIME_INCLUDE_DIR $SHERPA_ONNXRUNTIME_INCLUDE_DIR"

cmake -DCMAKE_TOOLCHAIN_FILE="$ANDROID_NDK/build/cmake/android.toolchain.cmake" \
    -DBUILD_PIPER_PHONMIZE_EXE=OFF \
    -DBUILD_PIPER_PHONMIZE_TESTS=OFF \
    -DBUILD_ESPEAK_NG_EXE=OFF \
    -DBUILD_ESPEAK_NG_TESTS=OFF \
    -DCMAKE_BUILD_TYPE=Release \
    -DBUILD_SHARED_LIBS=ON \
    -DSHERPA_ONNX_ENABLE_PYTHON=OFF \
    -DSHERPA_ONNX_ENABLE_TESTS=OFF \
    -DSHERPA_ONNX_ENABLE_CHECK=OFF \
    -DSHERPA_ONNX_ENABLE_PORTAUDIO=OFF \
    -DSHERPA_ONNX_ENABLE_JNI=ON \
    -DSHERPA_ONNX_ENABLE_C_API=OFF \
    -DCMAKE_INSTALL_PREFIX=./install \
    -DANDROID_ABI="arm64-v8a" \
    -DANDROID_PLATFORM=android-21 ..

# Please use -DANDROID_PLATFORM=android-27 if you want to use Android NNAPI

# make VERBOSE=1 -j4
make -j4
make install/strip
cp -fv $onnxruntime_version/jni/arm64-v8a/libonnxruntime.so install/lib
rm -rf install/lib/pkgconfig

# To run the generated binaries on Android, please use the following steps.
#
#
# 1. Copy sherpa-onnx and its dependencies to Android
#
#   cd build-android-arm64-v8a/install/lib
#   adb push ./lib*.so /data/local/tmp
#   cd ../bin
#   adb push ./sherpa-onnx /data/local/tmp
#
# 2. Login into Android
#
#   adb shell
#   cd /data/local/tmp
#   ./sherpa-onnx
#
# which shows the following error log:
#
#  CANNOT LINK EXECUTABLE "./sherpa-onnx": library "libsherpa-onnx-core.so" not found: needed by main executable
#
# Please run:
#
#  export LD_LIBRARY_PATH=$PWD:$LD_LIBRARY_PATH
#
# and then you can run:
#
#  ./sherpa-onnx
#
# It should show the help message of sherpa-onnx.
#
# Please use the above approach to copy model files to your phone.
