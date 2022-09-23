# Introduction

See <https://github.com/k2-fsa/sherpa>

This repo uses [onnxruntime](https://github.com/microsoft/onnxruntime) and
does not depend on libtorch.


# Onnxruntime Installation
```
git clone --recursive --branch v1.12.1 https://github.com/Microsoft/onnxruntime
cd onnxruntime
./build.sh \
    --config RelWithDebInfo \
    --build_shared_lib \
    --build_wheel \
    --skip_tests \
    --parallel 16
cd build/Linux/RelWithDebInfo
sudo make install
export LD_LIBRARY_PATH=/path/to/onnxruntime/build/Linux/RelWithDebInfo:$LD_LIBRARY_PATH
```

# Usage
```
git clone https://github.com/k2-fsa/sherpa-onnx
cd sherpa-onnx
mkdir build
cd build
cmake -DONNXRUNTIME_ROOTDIR=/path/to/onnxruntime \
      -DKALDI_NATIVE_IO_INSTALL_PREFIX=/path/to/kaldi_native_io ..
make
./bin/sherpa-onnx path/to/encoder.onnx \
                  path/to/decoder.onnx \
                  path/to/joiner.onnx \
                  path/to/joiner_encoder_proj.onnx \
                  path/to/joiner_decoder_proj.onnx \
                  path/to/tokens.txt \
                  greedy \
                  path/to/audio.wav
```
