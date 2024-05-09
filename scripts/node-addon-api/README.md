# Introduction

This folder contains `node-addon-api` wrapper for `sherpa-onnx`.

Caution: This folder is for developer only.

## Usage

```bash
git clone https://github.com/k2-fsa/sherpa-onnx
cd sherpa-onnx
mkdir build
cd build
cmake -DCMAKE_INSTALL_PREFIX=./install -DBUILD_SHARED_LIBS=ON ..
make -j install
export PKG_CONFIG_PATH=$PWD/install:$PKG_CONFIG_PATH
cd ../scripts/node-addon-api/

./node_modules/.bin/node-gyp build --verbose

# see test/test_asr_streaming_transducer.js
# for usages
```
