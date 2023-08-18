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

log "https://huggingface.co/csukuangfj/sherpa-onnx-streaming-zipformer-en-2023-06-26"

# Download the model
# see https://k2-fsa.github.io/sherpa/onnx/pretrained_models/online-transducer/zipformer-transducer-models.html#csukuangfj-sherpa-onnx-streaming-zipformer-en-2023-06-26-english
repo_url=https://huggingface.co/csukuangfj/sherpa-onnx-streaming-zipformer-en-2023-06-26
log "Start testing ${repo_url}"
repo=$(basename $repo_url)
log "Download pretrained model and test-data from $repo_url"
GIT_LFS_SKIP_SMUDGE=1 git clone $repo_url
pushd $repo
git lfs pull --include "*.onnx"

# remove .git to save spaces
rm -rf .git
rm README.md
rm -v *64*
rm -v encoder-epoch-99-avg-1-chunk-16-left-128.onnx
rm -v decoder-epoch-99-avg-1-chunk-16-left-128.int8.onnx
rm -v joiner-epoch-99-avg-1-chunk-16-left-128.int8.onnx
rm -rfv test_wavs
rm -v *.sh
ls -lh
popd

mv -v $repo ./android/SherpaOnnx/app/src/main/assets/
tree ./android/SherpaOnnx/app/src/main/assets/

pushd android/SherpaOnnx/app/src/main/java/com/k2fsa/sherpa/onnx
sed -i.bak s/"type = 0"/"type = 6"/ ./MainActivity.kt
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

  cp -v ./build-android-$src_arch/install/lib/*.so ./android/SherpaOnnx/app/src/main/jniLibs/$arch/

  pushd ./android/SherpaOnnx
  ./gradlew build
  popd

  mv android/SherpaOnnx/app/build/outputs/apk/debug/app-debug.apk ./apks/sherpa-onnx-${SHERPA_ONNX_VERSION}-$arch-en.apk
  ls -lh apks
  rm -v ./android/SherpaOnnx/app/src/main/jniLibs/$arch/*.so
done

git checkout .

rm -rf ./android/SherpaOnnx/app/src/main/assets/$repo

# type 3, encoder int8, zh
log "https://huggingface.co/pkufool/icefall-asr-zipformer-streaming-wenetspeech-20230615"
# see
# https://k2-fsa.github.io/sherpa/onnx/pretrained_models/online-transducer/zipformer-transducer-models.html#pkufool-icefall-asr-zipformer-streaming-wenetspeech-20230615-chinese

# Download the model
repo_url=https://huggingface.co/pkufool/icefall-asr-zipformer-streaming-wenetspeech-20230615
log "Start testing ${repo_url}"
repo=$(basename $repo_url)
log "Download pretrained model and test-data from $repo_url"
GIT_LFS_SKIP_SMUDGE=1 git clone $repo_url
pushd $repo
git lfs pull --include "*.onnx"

# remove .git to save spaces
rm -rf .git
rm README.md
rm -rf logs
rm -rf scripts
rm -rf test_wavs
rm -rf data/.DS_Store
rm data/lang_char/*.pt
rm data/lang_char/lexicon.txt
rm data/lang_char/words.txt
rm -v exp/*.pt
rm -v exp/*32*
rm -v exp/*256*
rm -v exp/encoder-epoch-12-avg-4-chunk-16-left-128.onnx
rm -v exp/decoder-epoch-12-avg-4-chunk-16-left-128.int8.onnx
rm -v exp/joiner-epoch-12-avg-4-chunk-16-left-128.int8.onnx
rm -rf exp/tensorboard

popd

mv -v $repo ./android/SherpaOnnx/app/src/main/assets/
tree ./android/SherpaOnnx/app/src/main/assets/
git checkout .

pushd android/SherpaOnnx/app/src/main/java/com/k2fsa/sherpa/onnx
sed -i.bak s/"type = 0"/"type = 3"/ ./MainActivity.kt
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

  cp -v ./build-android-$src_arch/install/lib/*.so ./android/SherpaOnnx/app/src/main/jniLibs/$arch/

  pushd ./android/SherpaOnnx
  ./gradlew build
  popd

  mv android/SherpaOnnx/app/build/outputs/apk/debug/app-debug.apk ./apks/sherpa-onnx-${SHERPA_ONNX_VERSION}-$arch-zh.apk
  ls -lh apks
  rm -v ./android/SherpaOnnx/app/src/main/jniLibs/$arch/*.so
done

git checkout .

rm -rf ./android/SherpaOnnx/app/src/main/assets/$repo

# type 7, encoder int8, french
log "https://huggingface.co/shaojieli/sherpa-onnx-streaming-zipformer-fr-2023-04-14"
# see
# https://k2-fsa.github.io/sherpa/onnx/pretrained_models/online-transducer/zipformer-transducer-models.html#shaojieli-sherpa-onnx-streaming-zipformer-fr-2023-04-14-french

# Download the model
repo_url=https://huggingface.co/shaojieli/sherpa-onnx-streaming-zipformer-fr-2023-04-14
log "Start testing ${repo_url}"
repo=$(basename $repo_url)
log "Download pretrained model and test-data from $repo_url"
GIT_LFS_SKIP_SMUDGE=1 git clone $repo_url
pushd $repo
git lfs pull --include "*.onnx"

# remove .git to save spaces
rm -rf .git
rm -v .gitattributes
rm -rf test_wavs
rm -v README.md
rm -v encoder-epoch-29-avg-9-with-averaged-model.onnx
rm -v decoder-epoch-29-avg-9-with-averaged-model.int8.onnx
rm -v joiner-epoch-29-avg-9-with-averaged-model.int8.onnx
rm -v export*.sh

popd

mv -v $repo ./android/SherpaOnnx/app/src/main/assets/
tree ./android/SherpaOnnx/app/src/main/assets/
git checkout .

pushd android/SherpaOnnx/app/src/main/java/com/k2fsa/sherpa/onnx
sed -i.bak s/"type = 0"/"type = 7"/ ./MainActivity.kt
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

  cp -v ./build-android-$src_arch/install/lib/*.so ./android/SherpaOnnx/app/src/main/jniLibs/$arch/

  pushd ./android/SherpaOnnx
  ./gradlew build
  popd

  mv android/SherpaOnnx/app/build/outputs/apk/debug/app-debug.apk ./apks/sherpa-onnx-${SHERPA_ONNX_VERSION}-$arch-fr.apk
  ls -lh apks
  rm -v ./android/SherpaOnnx/app/src/main/jniLibs/$arch/*.so
done

git checkout .

rm -rf ./android/SherpaOnnx/app/src/main/assets/$repo

# type 8, encoder int8, Binglual English + Chinese
log "https://huggingface.co/csukuangfj/sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20"
# see
# https://k2-fsa.github.io/sherpa/onnx/pretrained_models/online-transducer/zipformer-transducer-models.html#csukuangfj-sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20-bilingual-chinese-english

# Download the model
repo_url=https://huggingface.co/csukuangfj/sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20
log "Start testing ${repo_url}"
repo=$(basename $repo_url)
log "Download pretrained model and test-data from $repo_url"
GIT_LFS_SKIP_SMUDGE=1 git clone $repo_url
pushd $repo
git lfs pull --include "*.onnx"

# remove .git to save spaces
rm -rf .git
rm -v .gitattributes
rm -rf test_wavs
rm -v README.md
rm -v export*.sh
rm -v *state*
rm -v encoder-epoch-99-avg-1.onnx
rm -v decoder-epoch-99-avg-1.int8.onnx
rm -v joiner-epoch-99-avg-1.int8.onnx

popd

mv -v $repo ./android/SherpaOnnx/app/src/main/assets/
tree ./android/SherpaOnnx/app/src/main/assets/
git checkout .

pushd android/SherpaOnnx/app/src/main/java/com/k2fsa/sherpa/onnx
sed -i.bak s/"type = 0"/"type = 8"/ ./MainActivity.kt
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

  cp -v ./build-android-$src_arch/install/lib/*.so ./android/SherpaOnnx/app/src/main/jniLibs/$arch/

  pushd ./android/SherpaOnnx
  ./gradlew build
  popd

  mv android/SherpaOnnx/app/build/outputs/apk/debug/app-debug.apk ./apks/sherpa-onnx-${SHERPA_ONNX_VERSION}-$arch-bilingual-en-zh.apk
  ls -lh apks
  rm -v ./android/SherpaOnnx/app/src/main/jniLibs/$arch/*.so
done

git checkout .

rm -rf ./android/SherpaOnnx/app/src/main/assets/$repo

ls -lh apks/
