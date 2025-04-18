#!/usr/bin/env bash
set -ex

if [ x$BUILD_SHARED_LIBS == xOFF ]; then
  echo "BUILD_SHARED_LIBS=OFF is ignored for Android x86."
  echo "Always link with libonnxruntime.so"
  sleep 2
fi

dir=$PWD/build-android-x86

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

if [ ! -f $onnxruntime_version/jni/x86/libonnxruntime.so ]; then
  mkdir -p $onnxruntime_version
  pushd $onnxruntime_version
  wget -c -q https://github.com/csukuangfj/onnxruntime-libs/releases/download/v${onnxruntime_version}/onnxruntime-android-${onnxruntime_version}.zip
  unzip onnxruntime-android-${onnxruntime_version}.zip
  rm onnxruntime-android-${onnxruntime_version}.zip
  popd
fi

export SHERPA_ONNXRUNTIME_LIB_DIR=$dir/$onnxruntime_version/jni/x86/
export SHERPA_ONNXRUNTIME_INCLUDE_DIR=$dir/$onnxruntime_version/headers/

echo "SHERPA_ONNXRUNTIME_LIB_DIR: $SHERPA_ONNXRUNTIME_LIB_DIR"
echo "SHERPA_ONNXRUNTIME_INCLUDE_DIR $SHERPA_ONNXRUNTIME_INCLUDE_DIR"

if [ -z $SHERPA_ONNX_ENABLE_TTS ]; then
  SHERPA_ONNX_ENABLE_TTS=ON
fi

if [ -z $SHERPA_ONNX_ENABLE_SPEAKER_DIARIZATION ]; then
  SHERPA_ONNX_ENABLE_SPEAKER_DIARIZATION=ON
fi

if [ -z $SHERPA_ONNX_ENABLE_BINARY ]; then
  SHERPA_ONNX_ENABLE_BINARY=OFF
fi

if [ -z $SHERPA_ONNX_ENABLE_C_API ]; then
  SHERPA_ONNX_ENABLE_C_API=OFF
fi

if [ -z $SHERPA_ONNX_ENABLE_JNI ]; then
  SHERPA_ONNX_ENABLE_JNI=ON
fi

cmake -DCMAKE_TOOLCHAIN_FILE="$ANDROID_NDK/build/cmake/android.toolchain.cmake" \
    -DSHERPA_ONNX_ENABLE_TTS=$SHERPA_ONNX_ENABLE_TTS \
    -DSHERPA_ONNX_ENABLE_SPEAKER_DIARIZATION=$SHERPA_ONNX_ENABLE_SPEAKER_DIARIZATION \
    -DSHERPA_ONNX_ENABLE_BINARY=$SHERPA_ONNX_ENABLE_BINARY \
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
    -DSHERPA_ONNX_ENABLE_JNI=$SHERPA_ONNX_ENABLE_JNI \
    -DCMAKE_INSTALL_PREFIX=./install \
    -DANDROID_ABI="x86" \
    -DSHERPA_ONNX_ENABLE_C_API=$SHERPA_ONNX_ENABLE_C_API \
    -DSHERPA_ONNX_ENABLE_WEBSOCKET=OFF \
    -DANDROID_PLATFORM=android-21 ..

# make VERBOSE=1 -j4
make -j4
make install/strip
cp -fv $onnxruntime_version/jni/x86/libonnxruntime.so install/lib
rm -rf install/lib/pkgconfig

if [ -f install/lib/libsherpa-onnx-c-api.so ]; then
  cat >install/lib/README.md <<EOF
# Introduction

Note that if you use Android Studio, then you only need to
copy libonnxruntime.so and libsherpa-onnx-jni.so
to your jniLibs, and you don't need libsherpa-onnx-c-api.so or
libsherpa-onnx-cxx-api.so.

libsherpa-onnx-c-api.so and libsherpa-onnx-cxx-api.so are for users
who don't use JNI. In that case, libsherpa-onnx-jni.so is not needed.

In any case, libonnxruntime.is is always needed.
EOF
  ls -lh install/lib/README.md
fi
