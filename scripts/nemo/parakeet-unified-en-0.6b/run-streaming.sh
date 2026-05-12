#!/usr/bin/env bash
# Copyright      2026  Xiaomi Corp.        (authors: Fangjun Kuang)

set -ex

log() {
  # This function is from espnet
  local fname=${BASH_SOURCE[1]##*/}
  echo -e "$(date '+%Y-%m-%d %H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}

if [ ! -f ./parakeet-unified-en-0.6b.nemo ]; then
  curl -SL -O https://huggingface.co/nvidia/parakeet-unified-en-0.6b/resolve/main/parakeet-unified-en-0.6b.nemo
fi

curl -SL -O https://huggingface.co/nvidia/parakeet-unified-en-0.6b/resolve/main/bias.md
curl -SL -O https://huggingface.co/nvidia/parakeet-unified-en-0.6b/resolve/main/explainability.md
curl -SL -O https://huggingface.co/nvidia/parakeet-unified-en-0.6b/resolve/main/privacy.md
curl -SL -O https://huggingface.co/nvidia/parakeet-unified-en-0.6b/resolve/main/safety.md

curl -SL -O https://dldata-public.s3.us-east-2.amazonaws.com/2086-149220-0033.wav

pip install \
  "nemo_toolkit[asr] @ git+https://github.com/NVIDIA/NeMo.git" \
  "numpy<2" \
  ipython \
  kaldi-native-fbank \
  librosa \
  onnx \
  onnxruntime \
  soundfile

latency_list=(
1120ms
560ms
240ms
)

for latency in ${latency_list[@]}; do
  rm -fv layers.*.* Constant_* onnx__*
  ls -lh
  echo "---"
  ls -lh
  echo "---"
  python3 ./export_onnx_streaming.py --latency $latency

  ls -lh *.onnx

  echo "---int8----"
  python3 ./test_onnx_streaming.py \
    --encoder ./encoder.int8.onnx \
    --decoder ./decoder.int8.onnx \
    --joiner ./joiner.int8.onnx \
    --tokens ./tokens.txt \
    --wav 2086-149220-0033.wav

  echo "---fp32----"
  python3 ./test_onnx_streaming.py \
    --encoder ./encoder.int8.onnx \
    --decoder ./decoder.onnx \
    --joiner ./joiner.onnx \
    --tokens ./tokens.txt \
    --wav 2086-149220-0033.wav

  d=sherpa-onnx-nemo-parakeet-unified-en-0.6b-streaming-$latency
  mkdir -p $d
  mkdir -p $d/test_wavs
  mv -v encoder.onnx $d
  mv -v encoder.weights $d
  mv -v decoder.onnx $d
  mv -v joiner.onnx $d
  cp -v tokens.txt $d
  cp -v 2086-149220-0033.wav $d/test_wavs/0.wav
  cp -v bias.md $d
  cp -v explainability.md $d
  cp -v privacy.md $d
  cp -v safety.md $d
  echo "----$d---"
  ls -lh $d
  echo "----"

  d=sherpa-onnx-nemo-parakeet-unified-en-0.6b-int8-streaming-$latency
  mkdir -p $d
  mkdir -p $d/test_wavs
  mv -v encoder.int8.onnx $d
  mv -v decoder.int8.onnx $d
  mv -v joiner.int8.onnx $d
  mv -v tokens.txt $d
  cp -v 2086-149220-0033.wav $d/test_wavs/0.wav
  cp -v bias.md $d
  cp -v explainability.md $d
  cp -v privacy.md $d
  cp -v safety.md $d
  echo "----$d---"
  ls -lh $d
  tar cjfv $d.tar.bz2 $d
  ls -lh $d.tar.bz2
  echo "----"
done





