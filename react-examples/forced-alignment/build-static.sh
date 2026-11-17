#!/usr/bin/env bash
# Build the static bundle of the forced-alignment web app for one model.
#
# Usage:
#   ./build-static.sh <model_index> [output_dir]
#
# Prerequisites: the wasm package must exist, either
#   build-wasm-simd-forced-alignment/install/bin/wasm/forced-alignment/
# or public/sherpa/ must already contain the artifacts. See README.md.
#
# The output is a self-contained static site (html + css + js + the wasm
# package with the model files inside) ready for any static HTML server.

set -e

if [ $# -lt 1 ]; then
  echo "Usage: $0 <model_index> [output_dir]"
  echo "  model_index: 0=zh-aishell3 1=zh-biaobei 2=zh-large"
  echo "               3=en-v2.2.1 4=en-v3.0.0 5=en-v3.1.0"
  exit 1
fi

index=$1
out=${2:-dist-model-$index}

cd "$(dirname "$0")"

# 1. Copy the wasm package into public/sherpa/ if it is not there yet.
if [ ! -f public/sherpa/sherpa-onnx-wasm-main-fa.data ]; then
  src=../../build-wasm-simd-forced-alignment/install/bin/wasm/forced-alignment
  if [ ! -f $src/sherpa-onnx-wasm-main-fa.data ]; then
    echo "Cannot find the wasm package. Please run ./build-wasm-simd-forced-alignment.sh first."
    exit 1
  fi
  mkdir -p public/sherpa
  cp -v $src/sherpa-onnx-wasm-main-fa.{js,wasm,data} public/sherpa/
  cp -v ../../wasm/nodejs/sherpa-onnx-forced-aligner.js public/sherpa/
fi

# 2. Set the model index (the same sed the CI uses). The file is restored
# afterwards so building never dirties the working tree.
config=src/lib/model-config.ts
cp -v $config $config.orig
trap 'mv -f $config.orig $config' EXIT
sed -i.bak "s/export const SELECTED_MODEL_INDEX = .*/export const SELECTED_MODEL_INDEX = $index;/" $config
rm -f $config.bak
grep "SELECTED_MODEL_INDEX" $config

# 3. Build.
npm install --no-audit --no-fund
npm run build

# 4. Collect the static site.
rm -rf "$out"
mkdir -p "$out"
cp -rv dist/* "$out/"
echo "Static site is in $(pwd)/$out — serve it with any static HTTP server."
