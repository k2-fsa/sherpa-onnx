#!/usr/bin/env bash

# Please refer to
# https://k2-fsa.github.io/sherpa/onnx/pretrained_models/online-transducer/zipformer-transducer-models.html#csukuangfj-sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20-bilingual-chinese-english
# to download the model files
#
export LD_LIBRARY_PATH=$PWD:$LD_LIBRARY_PATH
export DYLD_LIBRARY_PATH=$PWD:$DYLD_LIBRARY_PATH

if [ ! -d ./icefall-asr-zipformer-streaming-wenetspeech-20230615 ]; then
  GIT_LFS_SKIP_SMUDGE=1 git clone https://huggingface.co/pkufool/icefall-asr-zipformer-streaming-wenetspeech-20230615
  cd icefall-asr-zipformer-streaming-wenetspeech-20230615
  git lfs pull --include "*.onnx"
  cd ..
fi

dotnet run -c Release \
  --tokens ./icefall-asr-zipformer-streaming-wenetspeech-20230615/data/lang_char/tokens.txt \
  --encoder ./icefall-asr-zipformer-streaming-wenetspeech-20230615/exp/encoder-epoch-12-avg-4-chunk-16-left-128.onnx \
  --decoder ./icefall-asr-zipformer-streaming-wenetspeech-20230615/exp/decoder-epoch-12-avg-4-chunk-16-left-128.onnx \
  --joiner ./icefall-asr-zipformer-streaming-wenetspeech-20230615/exp/joiner-epoch-12-avg-4-chunk-16-left-128.onnx
