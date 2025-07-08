#!/usr/bin/env bash

set -ex

python3 ./export-onnx-ctc.py

ls -lh *.onnx

mkdir -p test_wavs
pushd test_wavs
curl -SL -O https://huggingface.co/csukuangfj/reazonspeech-k2-v2-ja-en/resolve/main/test_wavs/transcripts.txt
curl -SL -O https://hf-mirror.com/csukuangfj/reazonspeech-k2-v2-ja-en/resolve/main/test_wavs/test_ja_1.wav
curl -SL -O https://hf-mirror.com/csukuangfj/reazonspeech-k2-v2-ja-en/resolve/main/test_wavs/test_ja_2.wav
popd

d=sherpa-onnx-nemo-parakeet-tdt_ctc-0.6b-ja-35000-int8

mkdir -p $d
mv -v model.int8.onnx $d/
cp -v tokens.txt $d/
cp -av test_wavs $d
ls -lh $d


d=sherpa-onnx-nemo-parakeet-tdt_ctc-0.6b-ja-35000-int8
python3 ./test-onnx-ctc-non-streaming.py \
  --model $d/model.int8.onnx \
  --tokens $d/tokens.txt \
  --wav $d/test_wavs/test_ja_1.wav

python3 ./test-onnx-ctc-non-streaming.py \
  --model $d/model.int8.onnx \
  --tokens $d/tokens.txt \
  --wav $d/test_wavs/test_ja_2.wav
