#!/usr/bin/env  bash

set -e

dir=build-vision-os
mkdir -p $dir
cd $dir
onnxruntime_version=1.20.2
onnxruntime_xros_arm64=onnxruntime-vision_os-xros_arm64-$onnxruntime_version/onnxruntime.framework

SHERPA_ONNX_HF=huggingface.co

if [ "$SHERPA_ONNX_HF_MIRROW" == true ]; then
    SHERPA_ONNX_HF=hf-mirror.com
fi

if [ ! -f $onnxruntime_xros_arm64/libonnxruntime.dylib ]; then
  wget -c https://$SHERPA_ONNX_HF/csukuangfj/onnxruntime-libs/resolve/main/onnxruntime-vision_os-xros_arm64-$onnxruntime_version.zip
  unzip onnxruntime-vision_os-xros_arm64-$onnxruntime_version.zip
  rm onnxruntime-vision_os-xros_arm64-$onnxruntime_version.zip
fi

# First, for simulator
echo "Building for visionOS (arm64)"

export SHERPA_ONNXRUNTIME_LIB_DIR=$PWD/$onnxruntime_xros_arm64
export SHERPA_ONNXRUNTIME_INCLUDE_DIR=$PWD/$onnxruntime_xros_arm64/Headers

echo "SHERPA_ONNXRUNTIME_LIB_DIR: $SHERPA_ONNXRUNTIME_LIB_DIR"
echo "SHERPA_ONNXRUNTIME_INCLUDE_DIR $SHERPA_ONNXRUNTIME_INCLUDE_DIR"

cmake \
  -DBUILD_PIPER_PHONMIZE_EXE=OFF \
  -DBUILD_PIPER_PHONMIZE_TESTS=OFF \
  -DBUILD_ESPEAK_NG_EXE=OFF \
  -DBUILD_ESPEAK_NG_TESTS=OFF \
  -S .. \
  -DCMAKE_TOOLCHAIN_FILE=./toolchains/ios.toolchain.cmake \
  -DPLATFORM=VISIONOS \
  -DENABLE_BITCODE=0 \
  -DENABLE_ARC=1 \
  -DENABLE_VISIBILITY=0 \
  -DCMAKE_BUILD_TYPE=Release \
  -DBUILD_SHARED_LIBS=ON \
  -DSHERPA_ONNX_ENABLE_PYTHON=OFF \
  -DSHERPA_ONNX_ENABLE_TESTS=OFF \
  -DSHERPA_ONNX_ENABLE_CHECK=OFF \
  -DSHERPA_ONNX_ENABLE_PORTAUDIO=OFF \
  -DSHERPA_ONNX_ENABLE_JNI=OFF \
  -DSHERPA_ONNX_ENABLE_C_API=ON \
  -DSHERPA_ONNX_ENABLE_WEBSOCKET=OFF \
  -DDEPLOYMENT_TARGET=13.0 \
  -B build/vision_os_arm64

cmake --build build/vision_os_arm64 -j 4 --verbose
