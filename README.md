# Introduction

See <https://github.com/k2-fsa/sherpa>

This repo uses [onnxruntime](https://github.com/microsoft/onnxruntime) and
does not depend on libtorch.

We provide exported models in onnx format and they can be downloaded using
the following links:

- English: <https://huggingface.co/csukuangfj/icefall-asr-librispeech-pruned-transducer-stateless3-2022-05-13>
- Chinese: <https://huggingface.co/luomingshuang/icefall_asr_wenetspeech_pruned_transducer_stateless2>

**NOTE**: We provide only non-streaming models at present.


**HINT**: The script for exporting the English model can be found at
<https://github.com/k2-fsa/icefall/blob/master/egs/librispeech/ASR/pruned_transducer_stateless3/export.py>

**HINT**: The script for exporting the Chinese model can be found at
<https://github.com/k2-fsa/icefall/blob/master/egs/wenetspeech/ASR/pruned_transducer_stateless2/export.py>

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

./build/bin/sherpa-onnx --help

./build/bin/sherpa-onnx \
  ./icefall-asr-librispeech-pruned-transducer-stateless3-2022-05-13/data/lang_bpe_500/tokens.txt \
  ./icefall-asr-librispeech-pruned-transducer-stateless3-2022-05-13/exp/onnx/encoder.onnx \
  ./icefall-asr-librispeech-pruned-transducer-stateless3-2022-05-13/exp/onnx/decoder.onnx \
  ./icefall-asr-librispeech-pruned-transducer-stateless3-2022-05-13/exp/onnx/joiner.onnx \
  ./icefall-asr-librispeech-pruned-transducer-stateless3-2022-05-13/exp/onnx/joiner_encoder_proj.onnx \
  ./icefall-asr-librispeech-pruned-transducer-stateless3-2022-05-13/exp/onnx/joiner_decoder_proj.onnx \
  ./icefall-asr-librispeech-pruned-transducer-stateless3-2022-05-13/test_wavs/1089-134686-0001.wav
```

## Download the pretrained model (Chinese)

**Caution**: You have to run `git lfs install`. Otherwise, you will be **SAD** later.

```bash
git lfs install
git clone https://huggingface.co/luomingshuang/icefall_asr_wenetspeech_pruned_transducer_stateless2

./build/bin/sherpa-onnx --help

./build/bin/sherpa-onnx \
  ./icefall_asr_wenetspeech_pruned_transducer_stateless2/data/lang_char/tokens.txt \
  ./icefall_asr_wenetspeech_pruned_transducer_stateless2/exp/encoder-epoch-10-avg-2.onnx \
  ./icefall_asr_wenetspeech_pruned_transducer_stateless2/exp/decoder-epoch-10-avg-2.onnx \
  ./icefall_asr_wenetspeech_pruned_transducer_stateless2/exp/joiner-epoch-10-avg-2.onnx \
  ./icefall_asr_wenetspeech_pruned_transducer_stateless2/exp/joiner_encoder_proj-epoch-10-avg-2.onnx \
  ./icefall_asr_wenetspeech_pruned_transducer_stateless2/exp/joiner_decoder_proj-epoch-10-avg-2.onnx \
  ./icefall_asr_wenetspeech_pruned_transducer_stateless2/test_wavs/DEV_T0000000000.wav
```
