#!/usr/bin/env bash

# Please refer to
# https://k2-fsa.github.io/sherpa/onnx/pretrained_models/online-transducer/zipformer-transducer-models.html#csukuangfj-sherpa-onnx-streaming-zipformer-en-2023-06-26-english
# to download the model
# before you run this script.
#
# You can switch to a different online model if you need

./streaming-decode-files \
  --encoder ./sherpa-onnx-streaming-zipformer-en-2023-06-26/encoder-epoch-99-avg-1-chunk-16-left-128.onnx \
  --decoder ./sherpa-onnx-streaming-zipformer-en-2023-06-26/decoder-epoch-99-avg-1-chunk-16-left-128.onnx \
  --joiner ./sherpa-onnx-streaming-zipformer-en-2023-06-26/joiner-epoch-99-avg-1-chunk-16-left-128.onnx \
  --tokens ./sherpa-onnx-streaming-zipformer-en-2023-06-26/tokens.txt \
  --model-type zipformer2 \
  --debug 0 \
  ./sherpa-onnx-streaming-zipformer-en-2023-06-26/test_wavs/0.wav
