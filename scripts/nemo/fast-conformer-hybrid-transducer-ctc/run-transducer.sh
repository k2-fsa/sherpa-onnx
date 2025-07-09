#!/usr/bin/env bash
# Copyright      2024  Xiaomi Corp.        (authors: Fangjun Kuang)

set -ex

if [ ! -e ./0.wav ]; then
  # curl -SL -O https://hf-mirror.com/csukuangfj/icefall-asr-librispeech-streaming-zipformer-small-2024-03-18/resolve/main/test_wavs/0.wav
  curl -SL -O https://huggingface.co/csukuangfj/icefall-asr-librispeech-streaming-zipformer-small-2024-03-18/resolve/main/test_wavs/0.wav
fi

ms=(
80
480
1040
)

for m in ${ms[@]}; do
  ./export-onnx-transducer.py --model $m
  d=sherpa-onnx-nemo-streaming-fast-conformer-transducer-en-${m}ms
  d_int8=sherpa-onnx-nemo-streaming-fast-conformer-transducer-en-${m}ms-int8
  if [ ! -f $d/encoder.onnx ]; then
    mkdir -p $d $d_int8
    mv -v encoder.onnx $d/
    mv -v decoder.onnx $d/
    mv -v joiner.onnx $d/
    cp -v tokens.txt $d/

    mv -v encoder.int8.onnx $d_int8/
    mv -v decoder.int8.onnx $d_int8/
    mv -v joiner.int8.onnx $d_int8/
    mv -v tokens.txt $d_int8/

    echo "---$d---"
    ls -lh $d

    echo "---$d_int8---"
    ls -lh $d_int8
  fi
done

# Now test the exported models

for m in ${ms[@]}; do
  d=sherpa-onnx-nemo-streaming-fast-conformer-transducer-en-${m}ms
  python3 ./test-onnx-transducer.py \
    --encoder $d/encoder.onnx \
    --decoder $d/decoder.onnx \
    --joiner $d/joiner.onnx \
    --tokens $d/tokens.txt \
    --wav ./0.wav

  d=sherpa-onnx-nemo-streaming-fast-conformer-transducer-en-${m}ms-int8
  python3 ./test-onnx-transducer.py \
    --encoder $d/encoder.int8.onnx \
    --decoder $d/decoder.int8.onnx \
    --joiner $d/joiner.int8.onnx \
    --tokens $d/tokens.txt \
    --wav ./0.wav
done
