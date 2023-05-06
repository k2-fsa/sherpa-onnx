#!/usr/bin/env bash
#
if [ ! -d ./sherpa-onnx-zipformer-en-2023-04-01 ]; then
  GIT_LFS_SKIP_SMUDGE=1 git clone https://huggingface.co/csukuangfj/sherpa-onnx-zipformer-en-2023-04-01
  cd sherpa-onnx-zipformer-en-2023-04-01
  git lfs pull --include "*.onnx"
  cd ..
fi

dotnet run \
  --tokens=./sherpa-onnx-zipformer-en-2023-04-01/tokens.txt \
  --encoder=./sherpa-onnx-zipformer-en-2023-04-01/encoder-epoch-99-avg-1.onnx \
  --decoder=./sherpa-onnx-zipformer-en-2023-04-01/decoder-epoch-99-avg-1.onnx \
  --joiner=./sherpa-onnx-zipformer-en-2023-04-01/joiner-epoch-99-avg-1.onnx \
  --num-threads=2 \
  --decoding-method=modified_beam_search \
  --files ./sherpa-onnx-zipformer-en-2023-04-01/test_wavs/0.wav \
  ./sherpa-onnx-zipformer-en-2023-04-01/test_wavs/1.wav \
  ./sherpa-onnx-zipformer-en-2023-04-01/test_wavs/8k.wav
