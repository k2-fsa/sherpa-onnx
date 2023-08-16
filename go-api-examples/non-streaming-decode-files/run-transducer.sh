#!/usr/bin/env bash

# Please refer to
# https://k2-fsa.github.io/sherpa/onnx/pretrained_models/offline-transducer/zipformer-transducer-models.html#csukuangfj-sherpa-onnx-zipformer-en-2023-06-26-english
# to download the model
# before you run this script.
#
# You can switch to a different offline model if you need

./non-streaming-decode-files \
  --encoder ./sherpa-onnx-zipformer-en-2023-06-26/encoder-epoch-99-avg-1.onnx \
  --decoder ./sherpa-onnx-zipformer-en-2023-06-26/decoder-epoch-99-avg-1.onnx \
  --joiner ./sherpa-onnx-zipformer-en-2023-06-26/joiner-epoch-99-avg-1.onnx \
  --tokens ./sherpa-onnx-zipformer-en-2023-06-26/tokens.txt \
  --model-type transducer \
  --debug 0 \
  ./sherpa-onnx-zipformer-en-2023-06-26/test_wavs/0.wav
