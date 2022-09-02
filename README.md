# Introduction

See <https://github.com/k2-fsa/sherpa>

This repo uses [onnxruntime](https://github.com/microsoft/onnxruntime) and
does not depend on libtorch.

# Usage

```bash
git clone https://github.com/k2-fsa/sherpa-onnx
cd sherpa-onnx
mkdir build
cd build
cmake ..
make -j

./bin/online-fbank-test
```
