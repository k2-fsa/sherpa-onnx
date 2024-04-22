#!/usr/bin/env bash

# Please set the environment variable ANDROID_NDK
# before running this script

# Inside the $ANDROID_NDK directory, you can find a binary ndk-build
# and some other files like the file "build/cmake/android.toolchain.cmake"

set -e

log() {
  # This function is from espnet
  local fname=${BASH_SOURCE[1]##*/}
  echo -e "$(date '+%Y-%m-%d %H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}

SHERPA_ONNX_VERSION=$(grep "SHERPA_ONNX_VERSION" ./CMakeLists.txt  | cut -d " " -f 2  | cut -d '"' -f 2)

log "Building APK for sherpa-onnx v${SHERPA_ONNX_VERSION}"

log "====================arm64-v8a================="
./build-android-arm64-v8a.sh
log "====================armv7-eabi================"
./build-android-armv7-eabi.sh
log "====================x86-64===================="
./build-android-x86-64.sh
log "====================x86===================="
./build-android-x86.sh


mkdir -p apks

log "https://github.com/snakers4/silero-vad/raw/master/files/silero_vad.onnx"

# Download the model
pushd ./android/SherpaOnnxVad/app/src/main/assets/
wget -c https://github.com/snakers4/silero-vad/raw/master/files/silero_vad.onnx
popd

for arch in arm64-v8a armeabi-v7a x86_64 x86; do
  log "------------------------------------------------------------"
  log "build apk for $arch"
  log "------------------------------------------------------------"
  src_arch=$arch
  if [ $arch == "armeabi-v7a" ]; then
    src_arch=armv7-eabi
  elif [ $arch == "x86_64" ]; then
    src_arch=x86-64
  fi

  ls -lh ./build-android-$src_arch/install/lib/*.so

  cp -v ./build-android-$src_arch/install/lib/*.so ./android/SherpaOnnxVad/app/src/main/jniLibs/$arch/

  pushd ./android/SherpaOnnxVad
  ./gradlew build
  popd

  mv android/SherpaOnnxVad/app/build/outputs/apk/debug/app-debug.apk ./apks/sherpa-onnx-${SHERPA_ONNX_VERSION}-$arch-silero-vad.apk
  ls -lh apks
  rm -v ./android/SherpaOnnxVad/app/src/main/jniLibs/$arch/*.so
done

rm -rf ./android/SherpaOnnxVad/app/src/main/assets/*.onnx

ls -lh apks/
