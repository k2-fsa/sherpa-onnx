#!/usr/bin/env bash

# Please refer to
# https://k2-fsa.github.io/sherpa/onnx/pretrained_models/online-transducer/zipformer-transducer-models.html#pkufool-icefall-asr-zipformer-streaming-wenetspeech-20230615-chinese
# to download the model
# before you run this script.
#
# You can switch to different online models if you need

./real-time-speech-recognition-from-microphone \
  --encoder ./icefall-asr-zipformer-streaming-wenetspeech-20230615/exp/encoder-epoch-12-avg-4-chunk-16-left-128.onnx \
  --decoder ./icefall-asr-zipformer-streaming-wenetspeech-20230615/exp/decoder-epoch-12-avg-4-chunk-16-left-128.onnx \
  --joiner ./icefall-asr-zipformer-streaming-wenetspeech-20230615/exp/joiner-epoch-12-avg-4-chunk-16-left-128.onnx \
  --tokens ./icefall-asr-zipformer-streaming-wenetspeech-20230615/data/lang_char/tokens.txt \
  --model-type zipformer2
