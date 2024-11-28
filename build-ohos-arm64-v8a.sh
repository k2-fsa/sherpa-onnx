#!/usr/bin/env bash
set -ex

dir=$PWD/build-ohos-arm64-v8a

mkdir -p $dir
cd $dir

# Please first download the commandline tools from
# https://developer.huawei.com/consumer/cn/download/
#
# Example filename on Linux: commandline-tools-linux-x64-5.0.5.200.zip
# You can also download it from https://hf-mirror.com/csukuangfj/harmonyos-commandline-tools/tree/main

# mkdir /star-fj/fangjun/software/huawei
# cd /star-fj/fangjun/software/huawei
# wget https://hf-mirror.com/csukuangfj/harmonyos-commandline-tools/resolve/main/commandline-tools-linux-x64-5.0.5.200.zip
# unzip commandline-tools-linux-x64-5.0.5.200.zip
# rm commandline-tools-linux-x64-5.0.5.200.zip
if [ -z $OHOS_SDK_NATIVE_DIR ]; then
  OHOS_SDK_NATIVE_DIR=/star-fj/fangjun/software/huawei/command-line-tools/sdk/default/openharmony/native/
  # You can find the following content inside OHOS_SDK_NATIVE_DIR
  # ls -lh /star-fj/fangjun/software/huawei/command-line-tools/sdk/default/openharmony/native/
  # total 524K
  # -rw-r--r--  1 kuangfangjun root 501K Jan  1  2001 NOTICE.txt
  # drwxr-xr-x  3 kuangfangjun root    0 Nov  6 22:36 build
  # drwxr-xr-x  3 kuangfangjun root    0 Nov  6 22:36 build-tools
  # -rw-r--r--  1 kuangfangjun root  371 Jan  1  2001 compatible_config.json
  # drwxr-xr-x  4 kuangfangjun root    0 Nov  6 22:36 docs
  # drwxr-xr-x 10 kuangfangjun root    0 Nov  6 22:36 llvm
  # -rw-r--r--  1 kuangfangjun root  16K Jan  1  2001 nativeapi_syscap_config.json
  # -rw-r--r--  1 kuangfangjun root 5.9K Jan  1  2001 ndk_system_capability.json
  # -rw-r--r--  1 kuangfangjun root  167 Jan  1  2001 oh-uni-package.json
  # drwxr-xr-x  3 kuangfangjun root    0 Nov  6 22:36 sysroot
fi

if [ ! -d $OHOS_SDK_NATIVE_DIR ]; then
  OHOS_SDK_NATIVE_DIR=/Users/fangjun/software/command-line-tools/sdk/default/openharmony/native
  # (py38) fangjuns-MacBook-Pro:software fangjun$ ls -lh command-line-tools/sdk/default/openharmony/native/
  # total 752
  # -rw-r--r--   1 fangjun  staff   341K Jan  1  2001 NOTICE.txt
  # drwxr-xr-x   3 fangjun  staff    96B Nov  6 21:17 build
  # drwxr-xr-x   3 fangjun  staff    96B Nov  6 21:18 build-tools
  # -rw-r--r--   1 fangjun  staff   371B Jan  1  2001 compatible_config.json
  # drwxr-xr-x  10 fangjun  staff   320B Nov  6 21:18 llvm
  # -rw-r--r--   1 fangjun  staff    16K Jan  1  2001 nativeapi_syscap_config.json
  # -rw-r--r--   1 fangjun  staff   5.9K Jan  1  2001 ndk_system_capability.json
  # -rw-r--r--   1 fangjun  staff   167B Jan  1  2001 oh-uni-package.json
  # drwxr-xr-x   3 fangjun  staff    96B Nov  6 21:17 sysroot
fi

if [ ! -d $OHOS_SDK_NATIVE_DIR ]; then
  echo "Please first download Command Line Tools for HarmonyOS"
  echo "See https://developer.huawei.com/consumer/cn/download/"
  echo "or"
  echo "https://hf-mirror.com/csukuangfj/harmonyos-commandline-tools/tree/main"
  exit 1
fi

if [ ! -f $OHOS_SDK_NATIVE_DIR/llvm/bin/aarch64-unknown-linux-ohos-clang ]; then
  echo "$OHOS_SDK_NATIVE_DIR/llvm/bin/aarch64-unknown-linux-ohos-clang does not exist"
  echo "Please first download Command Line Tools for HarmonyOS"
  echo "See https://developer.huawei.com/consumer/cn/download/"
  echo "or"
  echo "https://hf-mirror.com/csukuangfj/harmonyos-commandline-tools/tree/main"
  exit 1
fi

export PATH=$OHOS_SDK_NATIVE_DIR/build-tools/cmake/bin:$PATH
export PATH=$OHOS_SDK_NATIVE_DIR/llvm/bin:$PATH

OHOS_TOOLCHAIN_FILE=$OHOS_SDK_NATIVE_DIR/build/cmake/ohos.toolchain.cmake

if [ ! -f $OHOS_TOOLCHAIN_FILE ]; then
  echo "$OHOS_TOOLCHAIN_FILE does not exist"
  echo "Please first download Command Line Tools for HarmonyOS"
  exit 1
fi

sleep 1
onnxruntime_version=1.16.3
onnxruntime_dir=onnxruntime-ohos-arm64-v8a-$onnxruntime_version

if [ ! -f $onnxruntime_dir/lib/libonnxruntime.so ]; then
  # wget -c  https://github.com/csukuangfj/onnxruntime-libs/releases/download/v${onnxruntime_version}/$onnxruntime_dir.zip
  wget -c https://hf-mirror.com/csukuangfj/onnxruntime-libs/resolve/main/$onnxruntime_dir.zip
  unzip $onnxruntime_dir.zip
  rm $onnxruntime_dir.zip
fi

export SHERPA_ONNXRUNTIME_LIB_DIR=$dir/$onnxruntime_dir/lib
export SHERPA_ONNXRUNTIME_INCLUDE_DIR=$dir/$onnxruntime_dir/include

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

cmake \
    -DOHOS_ARCH=arm64-v8a \
    -DCMAKE_TOOLCHAIN_FILE=$OHOS_TOOLCHAIN_FILE \
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
    -DSHERPA_ONNX_ENABLE_JNI=OFF \
    -DSHERPA_ONNX_ENABLE_C_API=ON \
    -DCMAKE_INSTALL_PREFIX=./install \
    ..

# make VERBOSE=1 -j4
make -j2
make install/strip
cp -fv $onnxruntime_dir/lib/libonnxruntime.so install/lib

rm -rf install/share
rm -rf install/lib/pkgconfig

d=../harmony-os/SherpaOnnxHar/sherpa_onnx/src/main/cpp/libs/arm64-v8a
if [ -d $d ]; then
  cp -v install/lib/libsherpa-onnx-c-api.so $d/
  cp -v install/lib/libonnxruntime.so $d/
fi
