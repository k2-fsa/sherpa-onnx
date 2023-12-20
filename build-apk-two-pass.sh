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

log "Building two-pass APK for sherpa-onnx v${SHERPA_ONNX_VERSION}"

log "====================arm64-v8a================="
./build-android-arm64-v8a.sh
log "====================armv7-eabi================"
./build-android-armv7-eabi.sh
log "====================x86-64===================="
./build-android-x86-64.sh
log "====================x86===================="
./build-android-x86.sh

mkdir -p apks

log "Download 1st pass streaming model (English)"

# Download the model
# see https://k2-fsa.github.io/sherpa/onnx/pretrained_models/online-transducer/zipformer-transducer-models.html#csukuangfj-sherpa-onnx-streaming-zipformer-en-20m-2023-02-17-english
repo_url=https://huggingface.co/csukuangfj/sherpa-onnx-streaming-zipformer-en-20M-2023-02-17
log "$repo_url"

log "Start downloading ${repo_url}"
repo=$(basename $repo_url)
log "Download pretrained model from $repo_url"
GIT_LFS_SKIP_SMUDGE=1 git clone $repo_url
pushd $repo
git lfs pull --include "*.onnx"

# remove .git to save spaces
rm -rf .git
rm README.md
rm -rf test_wavs
rm .gitattributes
rm export-onnx*.sh

rm encoder-epoch-99-avg-1.onnx
rm decoder-epoch-99-avg-1.int8.onnx
rm joiner-epoch-99-avg-1.onnx

ls -lh
popd

mv -v $repo ./android/SherpaOnnx2Pass/app/src/main/assets/
tree ./android/SherpaOnnx2Pass/app/src/main/assets/
repo_1st=$repo

# https://k2-fsa.github.io/sherpa/onnx/pretrained_models/whisper/tiny.en.html
repo_url=https://huggingface.co/csukuangfj/sherpa-onnx-whisper-tiny.en
log "$repo_url"

log "Start downloading ${repo_url}"
repo=$(basename $repo_url)
log "Download pretrained model from $repo_url"
GIT_LFS_SKIP_SMUDGE=1 git clone $repo_url
pushd $repo
git lfs pull --include "*.onnx"

# remove .git to save spaces
rm -rf .git
rm -fv README.md
rm -rf test_wavs
rm .gitattributes

rm -fv *.ort
rm tiny.en-encoder.onnx
rm tiny.en-decoder.onnx

ls -lh
popd

mv -v $repo ./android/SherpaOnnx2Pass/app/src/main/assets/
tree ./android/SherpaOnnx2Pass/app/src/main/assets/
repo_2nd=$repo

pushd android/SherpaOnnx2Pass/app/src/main/java/com/k2fsa/sherpa/onnx
# sed -i.bak s/"firstType = 1"/"firstType = 1"/ ./MainActivity.kt
sed -i.bak s/"secondType = 1"/"secondType = 2"/ ./MainActivity.kt
git diff
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

  cp -v ./build-android-$src_arch/install/lib/*.so ./android/SherpaOnnx2Pass/app/src/main/jniLibs/$arch/

  pushd ./android/SherpaOnnx2Pass
  ./gradlew build
  popd

  mv android/SherpaOnnx2Pass/app/build/outputs/apk/debug/app-debug.apk ./apks/sherpa-onnx-${SHERPA_ONNX_VERSION}-$arch-en-2pass-whisper-tiny.en.apk
  ls -lh apks
  rm -v ./android/SherpaOnnx2Pass/app/src/main/jniLibs/$arch/*.so
done

git checkout .

rm -rf ./android/SherpaOnnx2Pass/app/src/main/assets/$repo_1st
rm -rf ./android/SherpaOnnx2Pass/app/src/main/assets/$repo_2nd

log "=================================================="
log "   two-pass Chinese                               "
log "=================================================="

log "Download 1st pass streaming model (Chinese)"
# https://k2-fsa.github.io/sherpa/onnx/pretrained_models/online-transducer/zipformer-transducer-models.html#sherpa-onnx-streaming-zipformer-zh-14m-2023-02-23
repo_url=https://huggingface.co/csukuangfj/sherpa-onnx-streaming-zipformer-zh-14M-2023-02-23
log "$repo_url"

log "Start downloading ${repo_url}"
repo=$(basename $repo_url)
log "Download pretrained model from $repo_url"
GIT_LFS_SKIP_SMUDGE=1 git clone $repo_url
pushd $repo
git lfs pull --include "*.onnx"

# remove .git to save spaces
rm -rf .git
rm README.md
rm -rf test_wavs
rm .gitattributes
rm export-onnx*.sh

rm encoder-epoch-99-avg-1.onnx
rm decoder-epoch-99-avg-1.int8.onnx
rm joiner-epoch-99-avg-1.onnx

ls -lh
popd

mv -v $repo ./android/SherpaOnnx2Pass/app/src/main/assets/
tree ./android/SherpaOnnx2Pass/app/src/main/assets/
repo_1st=$repo

# https://k2-fsa.github.io/sherpa/onnx/pretrained_models/offline-paraformer/paraformer-models.html#csukuangfj-sherpa-onnx-paraformer-zh-2023-03-28-chinese
repo_url=https://huggingface.co/csukuangfj/sherpa-onnx-paraformer-zh-2023-03-28
log "$repo_url"

log "Start downloading ${repo_url}"
repo=$(basename $repo_url)
log "Download pretrained model from $repo_url"
GIT_LFS_SKIP_SMUDGE=1 git clone $repo_url
pushd $repo
git lfs pull --include "*.onnx"

# remove .git to save spaces
rm -rf .git
rm README.md
rm -rf test_wavs
rm .gitattributes

rm model.onnx

ls -lh
popd

mv -v $repo ./android/SherpaOnnx2Pass/app/src/main/assets/
tree ./android/SherpaOnnx2Pass/app/src/main/assets/
repo_2nd=$repo

pushd android/SherpaOnnx2Pass/app/src/main/java/com/k2fsa/sherpa/onnx
sed -i.bak s/"firstType = 1"/"firstType = 0"/ ./MainActivity.kt
sed -i.bak s/"secondType = 1"/"secondType = 0"/ ./MainActivity.kt
git diff
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

  cp -v ./build-android-$src_arch/install/lib/*.so ./android/SherpaOnnx2Pass/app/src/main/jniLibs/$arch/

  pushd ./android/SherpaOnnx2Pass
  ./gradlew build
  popd

  mv android/SherpaOnnx2Pass/app/build/outputs/apk/debug/app-debug.apk ./apks/sherpa-onnx-${SHERPA_ONNX_VERSION}-$arch-zh-2pass-paraformer.apk
  ls -lh apks
  rm -v ./android/SherpaOnnx2Pass/app/src/main/jniLibs/$arch/*.so
done

git checkout .

rm -rf ./android/SherpaOnnx2Pass/app/src/main/assets/$repo_1st
rm -rf ./android/SherpaOnnx2Pass/app/src/main/assets/$repo_2nd

ls -lh apks/
