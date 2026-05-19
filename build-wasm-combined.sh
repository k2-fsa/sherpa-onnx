#!/bin/bash
#
# Copyright (c)  2024  Xiaomi Corporation

# Exit on error and print commands
set -ex

echo "=== Starting build process for sherpa-onnx WASM combined ==="

# Set environment flag to indicate we're using this script
export SHERPA_ONNX_IS_USING_BUILD_WASM_SH=1

# Create build directory
mkdir -p build-wasm-combined
cd build-wasm-combined

echo "=== Running CMake configuration ==="
# Configure with CMake
emcmake cmake \
  -DCMAKE_BUILD_TYPE=Release \
  -DSHERPA_ONNX_ENABLE_WASM=ON \
  -DSHERPA_ONNX_ENABLE_CHECK=OFF \
  -DSHERPA_ONNX_ENABLE_PORTAUDIO=OFF \
  -DSHERPA_ONNX_ENABLE_BINARY=OFF \
  -DSHERPA_ONNX_ENABLE_PYTHON=OFF \
  -DSHERPA_ONNX_ENABLE_JNI=OFF \
  -DSHERPA_ONNX_ENABLE_C_API=ON \
  -DSHERPA_ONNX_ENABLE_TEST=OFF \
  -DSHERPA_ONNX_ENABLE_WASM_COMBINED=ON \
  -DSHERPA_ONNX_INSTALL_TO_REPO=ON \
  ..

echo "=== Building the target ==="
# Build the target with full path to the target
emmake make -j $(nproc) sherpa-onnx-wasm-combined

echo "=== Installing the files ==="
# Install the files
emmake make install/strip

if [ $? -eq 0 ]; then
  echo "=== Build completed successfully! ==="
  echo "Files have been installed to bin/wasm/combined and copied to wasm/combined/"
else
  echo "=== Build failed! Check the error messages above ==="
  exit 1
fi 