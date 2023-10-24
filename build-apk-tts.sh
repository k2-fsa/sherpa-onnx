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

# Download the model
pushd ./android/SherpaOnnxTts/app/src/main/assets/
mkdir vits-vctk

cd vits-vctk
wget -qq https://huggingface.co/csukuangfj/vits-vctk/resolve/main/vits-vctk.onnx
wget -qq https://huggingface.co/csukuangfj/vits-vctk/resolve/main/lexicon.txt
wget -qq https://huggingface.co/csukuangfj/vits-vctk/resolve/main/tokens.txt
popd

for arch in arm64-v8a armeabi-v7a x86_64 x86; do
  log "------------------------------------------------------------"
  log "build tts apk for $arch"
  log "------------------------------------------------------------"
  src_arch=$arch
  if [ $arch == "armeabi-v7a" ]; then
    src_arch=armv7-eabi
  elif [ $arch == "x86_64" ]; then
    src_arch=x86-64
  fi

  ls -lh ./build-android-$src_arch/install/lib/*.so

  cp -v ./build-android-$src_arch/install/lib/*.so ./android/SherpaOnnxTts/app/src/main/jniLibs/$arch/

  pushd ./android/SherpaOnnxTts
  ./gradlew build
  popd

  mv android/SherpaOnnxTts/app/build/outputs/apk/debug/app-debug.apk ./apks/sherpa-onnx-${SHERPA_ONNX_VERSION}-$arch-en-tts-multi-speaker-vctk.apk
  ls -lh apks
  rm -v ./android/SherpaOnnxTts/app/src/main/jniLibs/$arch/*.so
done

rm -rf ./android/SherpaOnnxTts/app/src/main/assets/vits-vctk

git checkout .
pushd android/SherpaOnnxTts/app/src/main/java/com/k2fsa/sherpa/onnx
sed -i.bak s/"type = 0"/"type = 1"/ ./MainActivity.kt
git diff
popd

pushd ./android/SherpaOnnxTts/app/src/main/assets/
mkdir vits-zh-aishell3
cd vits-zh-aishell3

wget -qq https://huggingface.co/csukuangfj/vits-zh-aishell3/resolve/main/vits-aishell3.onnx
wget -qq https://huggingface.co/csukuangfj/vits-zh-aishell3/resolve/main/lexicon.txt
wget -qq  https://huggingface.co/csukuangfj/vits-zh-aishell3/resolve/main/tokens.txt

popd

for arch in arm64-v8a armeabi-v7a x86_64 x86; do
  log "------------------------------------------------------------"
  log "build tts apk for $arch"
  log "------------------------------------------------------------"
  src_arch=$arch
  if [ $arch == "armeabi-v7a" ]; then
    src_arch=armv7-eabi
  elif [ $arch == "x86_64" ]; then
    src_arch=x86-64
  fi

  ls -lh ./build-android-$src_arch/install/lib/*.so

  cp -v ./build-android-$src_arch/install/lib/*.so ./android/SherpaOnnxTts/app/src/main/jniLibs/$arch/

  pushd ./android/SherpaOnnxTts
  ./gradlew build
  popd

  mv android/SherpaOnnxTts/app/build/outputs/apk/debug/app-debug.apk ./apks/sherpa-onnx-${SHERPA_ONNX_VERSION}-$arch-zh-tts-multi-speaker-aishell3.apk
  ls -lh apks
  rm -v ./android/SherpaOnnxTts/app/src/main/jniLibs/$arch/*.so
done

rm -rf ./android/SherpaOnnxTts/app/src/main/assets/vits-vctk

git checkout .

ls -lh apks/
