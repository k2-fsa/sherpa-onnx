#!/usr/bin/env bash

set -ex
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/common.sh"
repo_dir="$(cd "$SCRIPT_DIR/../.." && pwd)"
test_dir="$(mktemp -d)"
trap 'rm -rf "$test_dir"' EXIT
cd "$test_dir"

download_and_extract https://github.com/k2-fsa/sherpa-onnx/releases/download/source-separation-models/sherpa-onnx-spleeter-2stems-fp16.tar.bz2
download https://github.com/k2-fsa/sherpa-onnx/releases/download/source-separation-models/qi-feng-le-zh.wav

node --expose-gc "$repo_dir/scripts/node-addon-api/test/test_source_separation.js" spleeter

download https://github.com/k2-fsa/sherpa-onnx/releases/download/source-separation-models/UVR-MDX-NET-Voc_FT.onnx
node --expose-gc "$repo_dir/scripts/node-addon-api/test/test_source_separation.js" uvr
