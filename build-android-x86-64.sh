#!/usr/bin/env bash
set -ex

dir=$PWD/build-android-x86-64

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
  ANDROID_NDK=/ceph-fj/fangjun/software/android-sdk/ndk/21.0.6113669
  # or use
  # ANDROID_NDK=/ceph-fj/fangjun/software/android-ndk
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

onnxruntime_version=v1.16.3

if [ ! -f ./android-onnxruntime-libs/$onnxruntime_version/jni/x86_64/libonnxruntime.so ]; then
  if [ ! -d android-onnxruntime-libs ]; then
    GIT_LFS_SKIP_SMUDGE=1 git clone https://huggingface.co/csukuangfj/android-onnxruntime-libs
  fi
  pushd android-onnxruntime-libs
  git lfs pull --include "$onnxruntime_version/jni/x86_64/libonnxruntime.so"
  ln -s $onnxruntime_version/jni .
  ln -s $onnxruntime_version/headers .
  popd
fi

ls -l ./android-onnxruntime-libs/jni/x86_64/libonnxruntime.so

# check filesize
filesize=$(ls -l ./android-onnxruntime-libs/jni/x86_64/libonnxruntime.so  | tr -s " " " " | cut -d " " -f 5)
if (( $filesize < 1000 )); then
  ls -lh ./android-onnxruntime-libs/jni/x86_64/libonnxruntime.so
  echo "Please use: git lfs pull to download libonnxruntime.so"
  exit 1
fi

export SHERPA_ONNXRUNTIME_LIB_DIR=$dir/android-onnxruntime-libs/jni/x86_64/
export SHERPA_ONNXRUNTIME_INCLUDE_DIR=$dir/android-onnxruntime-libs/headers/

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
    -DCMAKE_INSTALL_PREFIX=./install \
    -DANDROID_ABI="x86_64" \
    -DSHERPA_ONNX_ENABLE_C_API=OFF \
    -DSHERPA_ONNX_ENABLE_WEBSOCKET=OFF \
    -DANDROID_PLATFORM=android-21 ..

# make VERBOSE=1 -j4
make -j4
make install/strip
cp -fv android-onnxruntime-libs/jni/x86_64/libonnxruntime.so install/lib
