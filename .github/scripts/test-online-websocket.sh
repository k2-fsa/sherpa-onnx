#!/usr/bin/env bash

# set -e

log() {
  # This function is from espnet
  local fname=${BASH_SOURCE[1]##*/}
  echo -e "$(date '+%Y-%m-%d %H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}

echo "SERVER_EXE is $SERVER_EXE"
echo "CLIENT_EXE is $CLIENT_EXE"
echo "PATH: $PATH"

which $SERVER_EXE
which $CLIENT_EXE

repo_url=https://huggingface.co/csukuangfj/sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20
log "Start testing ${repo_url}"
repo=$(basename $repo_url)
log "Download pretrained model and test-data from $repo_url"

GIT_LFS_SKIP_SMUDGE=1 git clone $repo_url
pushd $repo
git lfs pull --include "*.onnx"
popd


log "starting the server"

$SERVER_EXE \
  --port=6008 \
  --num-work-threads=2 \
  --num-io-threads=1 \
  --tokens=./sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20/tokens.txt \
  --encoder=./sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20/encoder-epoch-99-avg-1.onnx \
  --decoder=./sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20/decoder-epoch-99-avg-1.onnx \
  --joiner=./sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20/joiner-epoch-99-avg-1.onnx \
  --log-file=./log.txt \
  --max-batch-size=5 \
  --loop-interval-ms=20 &

log "sleep 10 seconds to wait the server to start"

sleep 10

n=10
log "Start $n clients"

for i in $(seq 0 $n); do
  k=$(expr $i % 5)
  log "starting client $i, processing ${k}.wav"
  $CLIENT_EXE \
    --seconds-per-message=0.1 \
    --server-port=6008 \
    ./sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20/test_wavs/${k}.wav &
done

wait

echo "done"
