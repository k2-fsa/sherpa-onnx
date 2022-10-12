# Introduction

See <https://github.com/k2-fsa/sherpa>

This repo uses [onnxruntime](https://github.com/microsoft/onnxruntime) and
does not depend on libtorch.

We provide exported models in onnx format and they can be downloaded using
the following links:

- English: <https://huggingface.co/csukuangfj/icefall-asr-librispeech-pruned-transducer-stateless3-2022-05-13>
- Chinese: `TODO`

**NOTE**: We provide only non-streaming models at present.


# Usage

```bash
git clone https://github.com/k2-fsa/sherpa-onnx
cd sherpa-onnx
mkdir build
cd build
cmake ..
make -j6
cd ..
```

## Download the pretrained model (English)

**Caution**: You have to run `git lfs install`. Otherwise, you will be **SAD** later.

```bash
git lfs install
git clone https://huggingface.co/csukuangfj/icefall-asr-librispeech-pruned-transducer-stateless3-2022-05-13

./build/bin/sherpa-onnx \
  ./icefall-asr-librispeech-pruned-transducer-stateless3-2022-05-13/exp/onnx/encoder.onnx \
  ./icefall-asr-librispeech-pruned-transducer-stateless3-2022-05-13/exp/onnx/decoder.onnx \
  ./icefall-asr-librispeech-pruned-transducer-stateless3-2022-05-13/exp/onnx/joiner.onnx \
  ./icefall-asr-librispeech-pruned-transducer-stateless3-2022-05-13/exp/onnx/joiner_encoder_proj.onnx \
  ./icefall-asr-librispeech-pruned-transducer-stateless3-2022-05-13/exp/onnx/joiner_decoder_proj.onnx \
  ./icefall-asr-librispeech-pruned-transducer-stateless3-2022-05-13/data/lang_bpe_500/tokens.txt \
  greedy \
  ./icefall-asr-librispeech-pruned-transducer-stateless3-2022-05-13/test_wavs/1089-134686-0001.wav
```
