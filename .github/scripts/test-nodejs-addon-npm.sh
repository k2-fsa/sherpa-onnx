#!/usr/bin/env bash

set -ex

d=nodejs-addon-examples
echo "dir: $d"
cd $d
npm install --verbose
git status
ls -lh
ls -lh node_modules

export DYLD_LIBRARY_PATH=$PWD/node_modules/sherpa-onnx-darwin-x64:$DYLD_LIBRARY_PATH
export DYLD_LIBRARY_PATH=$PWD/node_modules/sherpa-onnx-darwin-arm64:$DYLD_LIBRARY_PATH
export LD_LIBRARY_PATH=$PWD/node_modules/sherpa-onnx-linux-x64:$LD_LIBRARY_PATH

curl -SL -O https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20.tar.bz2
tar xvf sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20.tar.bz2
rm sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20.tar.bz2

node test_asr_streaming_transducer.js

rm -rf sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20
