#!/usr/bin/env bash

set -e

log() {
  # This function is from espnet
  local fname=${BASH_SOURCE[1]##*/}
  echo -e "$(date '+%Y-%m-%d %H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}

export GIT_CLONE_PROTECTION_ACTIVE=false

echo "EXE is $EXE"
echo "PATH: $PATH"

which $EXE

log "------------------------------------------------------------"
log "Run NeMo transducer (English)"
log "------------------------------------------------------------"
repo_url=https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-nemo-streaming-fast-conformer-transducer-en-80ms.tar.bz2
curl -SL -O $repo_url
tar xvf sherpa-onnx-nemo-streaming-fast-conformer-transducer-en-80ms.tar.bz2
rm sherpa-onnx-nemo-streaming-fast-conformer-transducer-en-80ms.tar.bz2
repo=sherpa-onnx-nemo-streaming-fast-conformer-transducer-en-80ms

log "Start testing ${repo_url}"

waves=(
$repo/test_wavs/0.wav
$repo/test_wavs/1.wav
$repo/test_wavs/8k.wav
)

for wave in ${waves[@]}; do
  time $EXE \
  --tokens=$repo/tokens.txt \
  --encoder=$repo/encoder.onnx \
  --decoder=$repo/decoder.onnx \
  --joiner=$repo/joiner.onnx \
  --num-threads=2 \
  $wave
done

time $EXE \
  --tokens=$repo/tokens.txt \
  --encoder=$repo/encoder.onnx \
  --decoder=$repo/decoder.onnx \
  --joiner=$repo/joiner.onnx \
  --num-threads=2 \
  $repo/test_wavs/0.wav \
  $repo/test_wavs/1.wav \
  $repo/test_wavs/8k.wav

rm -rf $repo

log "------------------------------------------------------------"
log "Run LSTM transducer (English)"
log "------------------------------------------------------------"

repo_url=https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-lstm-en-2023-02-17.tar.bz2
curl -SL -O $repo_url
tar xvf sherpa-onnx-lstm-en-2023-02-17.tar.bz2
rm sherpa-onnx-lstm-en-2023-02-17.tar.bz2
repo=sherpa-onnx-lstm-en-2023-02-17

log "Start testing ${repo_url}"

waves=(
$repo/test_wavs/0.wav
$repo/test_wavs/1.wav
$repo/test_wavs/8k.wav
)

for wave in ${waves[@]}; do
  time $EXE \
  --tokens=$repo/tokens.txt \
  --encoder=$repo/encoder-epoch-99-avg-1.onnx \
  --decoder=$repo/decoder-epoch-99-avg-1.onnx \
  --joiner=$repo/joiner-epoch-99-avg-1.onnx \
  --num-threads=2 \
  $wave
done

for wave in ${waves[@]}; do
  time $EXE \
  --tokens=$repo/tokens.txt \
  --encoder=$repo/encoder-epoch-99-avg-1.int8.onnx \
  --decoder=$repo/decoder-epoch-99-avg-1.int8.onnx \
  --joiner=$repo/joiner-epoch-99-avg-1.int8.onnx \
  --num-threads=2 \
  $wave
done

rm -rf $repo

log "------------------------------------------------------------"
log "Run LSTM transducer (Chinese)"
log "------------------------------------------------------------"

repo_url=https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-lstm-zh-2023-02-20.tar.bz2
curl -SL -O $repo_url
tar xvf sherpa-onnx-lstm-zh-2023-02-20.tar.bz2
rm sherpa-onnx-lstm-zh-2023-02-20.tar.bz2
repo=sherpa-onnx-lstm-zh-2023-02-20

log "Start testing ${repo_url}"

waves=(
$repo/test_wavs/0.wav
$repo/test_wavs/1.wav
$repo/test_wavs/8k.wav
)

for wave in ${waves[@]}; do
  time $EXE \
  --tokens=$repo/tokens.txt \
  --encoder=$repo/encoder-epoch-11-avg-1.onnx \
  --decoder=$repo/decoder-epoch-11-avg-1.onnx \
  --joiner=$repo/joiner-epoch-11-avg-1.onnx \
  --num-threads=2 \
  $wave
done

for wave in ${waves[@]}; do
  time $EXE \
  --tokens=$repo/tokens.txt \
  --encoder=$repo/encoder-epoch-11-avg-1.int8.onnx \
  --decoder=$repo/decoder-epoch-11-avg-1.int8.onnx \
  --joiner=$repo/joiner-epoch-11-avg-1.int8.onnx \
  --num-threads=2 \
  $wave
done

rm -rf $repo

log "------------------------------------------------------------"
log "Run streaming Zipformer transducer (English)"
log "------------------------------------------------------------"

repo_url=https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-streaming-zipformer-en-2023-02-21.tar.bz2
curl -SL -O $repo_url
tar xvf sherpa-onnx-streaming-zipformer-en-2023-02-21.tar.bz2
rm sherpa-onnx-streaming-zipformer-en-2023-02-21.tar.bz2
repo=sherpa-onnx-streaming-zipformer-en-2023-02-21

log "Start testing ${repo_url}"

waves=(
$repo/test_wavs/0.wav
$repo/test_wavs/1.wav
$repo/test_wavs/8k.wav
)

for wave in ${waves[@]}; do
  time $EXE \
  --tokens=$repo/tokens.txt \
  --encoder=$repo/encoder-epoch-99-avg-1.onnx \
  --decoder=$repo/decoder-epoch-99-avg-1.onnx \
  --joiner=$repo/joiner-epoch-99-avg-1.onnx \
  --num-threads=2 \
  $wave
done

# test int8
#
for wave in ${waves[@]}; do
  time $EXE \
  --tokens=$repo/tokens.txt \
  --encoder=$repo/encoder-epoch-99-avg-1.int8.onnx \
  --decoder=$repo/decoder-epoch-99-avg-1.int8.onnx \
  --joiner=$repo/joiner-epoch-99-avg-1.int8.onnx \
  --num-threads=2 \
  $wave
done

rm -rf $repo

log "------------------------------------------------------------"
log "Run streaming Zipformer transducer (Bilingual, Chinese + English)"
log "------------------------------------------------------------"

repo_url=https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20.tar.bz2
curl -SL -O $repo_url
tar xvf sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20.tar.bz2
rm sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20.tar.bz2
repo=sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20

log "Start testing ${repo_url}"

waves=(
$repo/test_wavs/0.wav
$repo/test_wavs/1.wav
$repo/test_wavs/2.wav
$repo/test_wavs/3.wav
$repo/test_wavs/8k.wav
)

for wave in ${waves[@]}; do
  time $EXE \
  --tokens=$repo/tokens.txt \
  --encoder=$repo/encoder-epoch-99-avg-1.onnx \
  --decoder=$repo/decoder-epoch-99-avg-1.onnx \
  --joiner=$repo/joiner-epoch-99-avg-1.onnx \
  --num-threads=2 \
  $wave
done

for wave in ${waves[@]}; do
  time $EXE \
  --tokens=$repo/tokens.txt \
  --encoder=$repo/encoder-epoch-99-avg-1.int8.onnx \
  --decoder=$repo/decoder-epoch-99-avg-1.int8.onnx \
  --joiner=$repo/joiner-epoch-99-avg-1.int8.onnx \
  --num-threads=2 \
  $wave
done

# Decode a URL
if [ $EXE == "sherpa-onnx-ffmpeg" ]; then
  time $EXE \
  $repo/tokens.txt \
  $repo/encoder-epoch-99-avg-1.onnx \
  $repo/decoder-epoch-99-avg-1.onnx \
  $repo/joiner-epoch-99-avg-1.onnx \
  https://huggingface.co/csukuangfj/sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20/resolve/main/test_wavs/4.wav \
  2
fi

if [ $EXE == "sherpa-onnx-ffmpeg" ]; then
  time $EXE \
  $repo/tokens.txt \
  $repo/encoder-epoch-99-avg-1.int8.onnx \
  $repo/decoder-epoch-99-avg-1.int8.onnx \
  $repo/joiner-epoch-99-avg-1.int8.onnx \
  https://huggingface.co/csukuangfj/sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20/resolve/main/test_wavs/4.wav \
  2
fi

rm -rf $repo

log "------------------------------------------------------------"
log "Run streaming Conformer transducer (English)"
log "------------------------------------------------------------"

repo_url=https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-streaming-conformer-en-2023-05-09.tar.bz2
curl -SL -O $repo_url
tar xvf sherpa-onnx-streaming-conformer-en-2023-05-09.tar.bz2
rm sherpa-onnx-streaming-conformer-en-2023-05-09.tar.bz2
repo=sherpa-onnx-streaming-conformer-en-2023-05-09

log "Start testing ${repo_url}"

waves=(
$repo/test_wavs/0.wav
$repo/test_wavs/1.wav
$repo/test_wavs/2.wav
)

for wave in ${waves[@]}; do
  time $EXE \
  --tokens=$repo/tokens.txt \
  --encoder=$repo/encoder-epoch-99-avg-1.onnx \
  --decoder=$repo/decoder-epoch-99-avg-1.onnx \
  --joiner=$repo/joiner-epoch-99-avg-1.onnx \
  --num-threads=2 \
  $wave
done

for wave in ${waves[@]}; do
  time $EXE \
  --tokens=$repo/tokens.txt \
  --encoder=$repo/encoder-epoch-99-avg-1.int8.onnx \
  --decoder=$repo/decoder-epoch-99-avg-1.int8.onnx \
  --joiner=$repo/joiner-epoch-99-avg-1.int8.onnx \
  --num-threads=2 \
  $wave
done

rm -rf $repo
