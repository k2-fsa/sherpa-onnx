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

log "Building keyword spotting APK for sherpa-onnx v${SHERPA_ONNX_VERSION}"

log "====================arm64-v8a================="
./build-android-arm64-v8a.sh
log "====================armv7-eabi================"
#./build-android-armv7-eabi.sh
log "====================x86-64===================="
#./build-android-x86-64.sh
log "====================x86===================="
#./build-android-x86.sh


mkdir -p apks

log "https://www.modelscope.cn/pkufool/sherpa-onnx-kws-zipformer-wenetspeech-3.3M-2024-01-01.git"

# Download the model
repo_url=https://www.modelscope.cn/pkufool/sherpa-onnx-kws-zipformer-wenetspeech-3.3M-2024-01-01.git
#log "Start testing ${repo_url}"
##repo=$(basename $repo_url)
#repo=sherpa-onnx-kws-zipformer-wenetspeech-3.3M-2024-01-01
#log "Download pretrained model and test-data from $repo_url"
#GIT_LFS_SKIP_SMUDGE=1 git clone $repo_url
#pushd $repo
#git lfs pull --include "*.onnx"
#
## remove .git to save spaces
#rm -rf .git
#rm README.md
#rm -rfv test_wavs
#ls -lh
#popd
#
#mv -v $repo ./android/SherpaOnnxKws/app/src/main/assets/
tree ./android/SherpaOnnxKws/app/src/main/assets/

pushd android/SherpaOnnxKws/app/src/main/java/com/k2fsa/sherpa/onnx
#sed -i.bak s/"type = 0"/"type = 6"/ ./MainActivity.kt
#git diff
popd

#for arch in arm64-v8a armeabi-v7a x86_64 x86; do
for arch in arm64-v8a; do
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

  cp -v ./build-android-$src_arch/install/lib/*.so ./android/SherpaOnnxKws/app/src/main/jniLibs/$arch/

  pushd ./android/SherpaOnnxKws
  ./gradlew build
  popd

  mv android/SherpaOnnxKws/app/build/outputs/apk/debug/app-debug.apk ./apks/sherpa-onnx-${SHERPA_ONNX_VERSION}-$arch.apk
  ls -lh apks
  #rm -v ./android/SherpaOnnxKws/app/src/main/jniLibs/$arch/*.so
done

#git checkout .

#rm -rf ./android/SherpaOnnxKws/app/src/main/assets/$repo
