#!/usr/bin/env bash
# Copyright      2024  Xiaomi Corp.        (authors: Fangjun Kuang)

set -ex

log() {
  # This function is from espnet
  local fname=${BASH_SOURCE[1]##*/}
  echo -e "$(date '+%Y-%m-%d %H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}

if [ ! -e ./0.wav ]; then
  # curl -SL -O https://hf-mirror.com/csukuangfj/icefall-asr-librispeech-streaming-zipformer-small-2024-03-18/resolve/main/test_wavs/0.wav
  curl -SL -O https://huggingface.co/csukuangfj/icefall-asr-librispeech-streaming-zipformer-small-2024-03-18/resolve/main/test_wavs/0.wav
fi


# 24500 hours of English speech
url=https://catalog.ngc.nvidia.com/orgs/nvidia/teams/nemo/models/stt_en_fastconformer_ctc_large
name=$(basename https://catalog.ngc.nvidia.com/orgs/nvidia/teams/nemo/models/stt_en_fastconformer_ctc_large)
log "Process $name at $url"
./export-onnx-ctc.py --model $name
d=sherpa-onnx-nemo-fast-conformer-ctc-en-24500
mkdir -p $d
mv -v model.onnx $d/
mv -v tokens.txt $d/
ls -lh $d

# python3 ./test-onnx-ctc.py \
#   --model $d/model.onnx \
#   --tokens $d/tokens.txt \
#   --wav ./0.wav
