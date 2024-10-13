#!/usr/bin/env bash
#
# Please set the environment variable ANDROID_NDK
# before running this script

# Inside the $ANDROID_NDK directory, you can find a binary ndk-build
# and some other files like the file "build/cmake/android.toolchain.cmake"

set -ex

log() {
  # This function is from espnet
  local fname=${BASH_SOURCE[1]##*/}
  echo -e "$(date '+%Y-%m-%d %H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}

SHERPA_ONNX_VERSION=$(grep "SHERPA_ONNX_VERSION" ./CMakeLists.txt  | cut -d " " -f 2  | cut -d '"' -f 2)

log "Building Speaker identification APK for sherpa-onnx v${SHERPA_ONNX_VERSION}"

export SHERPA_ONNX_ENABLE_TTS=OFF

log "====================arm64-v8a================="
./build-android-arm64-v8a.sh
log "====================armv7-eabi================"
./build-android-armv7-eabi.sh
log "====================x86-64===================="
./build-android-x86-64.sh
log "====================x86===================="
./build-android-x86.sh

mkdir -p apks

pushd ./android/SherpaOnnxSpeakerDiarization/app/src/main/assets/

curl -SL -O https://github.com/k2-fsa/sherpa-onnx/releases/download/speaker-segmentation-models/sherpa-onnx-pyannote-segmentation-3-0.tar.bz2
tar xvf sherpa-onnx-pyannote-segmentation-3-0.tar.bz2
rm sherpa-onnx-pyannote-segmentation-3-0.tar.bz2
mv sherpa-onnx-pyannote-segmentation-3-0/model.onnx segmentation.onnx
rm -rf sherpa-onnx-pyannote-segmentation-3-0

curl -SL -O https://github.com/k2-fsa/sherpa-onnx/releases/download/speaker-recongition-models/3dspeaker_speech_eres2net_base_sv_zh-cn_3dspeaker_16k.onnx

echo "pwd: $PWD"
ls -lh

popd

for arch in arm64-v8a armeabi-v7a x86_64 x86; do
  log "------------------------------------------------------------"
  log "build speaker diarization apk for $arch"
  log "------------------------------------------------------------"
  src_arch=$arch
  if [ $arch == "armeabi-v7a" ]; then
    src_arch=armv7-eabi
  elif [ $arch == "x86_64" ]; then
    src_arch=x86-64
  fi

  ls -lh ./build-android-$src_arch/install/lib/*.so

  cp -v ./build-android-$src_arch/install/lib/*.so ./android/SherpaOnnxSpeakerDiarization/app/src/main/jniLibs/$arch/

  pushd ./android/SherpaOnnxSpeakerDiarization
  ./gradlew build
  popd

  mv android/SherpaOnnxSpeakerDiarization/app/build/outputs/apk/debug/app-debug.apk ./apks/sherpa-onnx-${SHERPA_ONNX_VERSION}-$arch-speaker-diarization-pyannote_audio-3dspeaker.apk
  ls -lh apks
  rm -v ./android/SherpaOnnxSpeakerDiarization/app/src/main/jniLibs/$arch/*.so
done

ls -lh apks
