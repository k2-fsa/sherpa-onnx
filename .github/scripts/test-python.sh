#!/usr/bin/env bash

set -e

log() {
  # This function is from espnet
  local fname=${BASH_SOURCE[1]##*/}
  echo -e "$(date '+%Y-%m-%d %H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}

wenet_models=(
sherpa-onnx-zh-wenet-aishell
sherpa-onnx-zh-wenet-aishell2
sherpa-onnx-zh-wenet-wenetspeech
sherpa-onnx-zh-wenet-multi-cn
sherpa-onnx-en-wenet-librispeech
sherpa-onnx-en-wenet-gigaspeech
)

mkdir -p /tmp/icefall-models
dir=/tmp/icefall-models

for name in ${wenet_models[@]}; do
  repo_url=https://huggingface.co/csukuangfj/$name
  log "Start testing ${repo_url}"
  repo=$dir/$(basename $repo_url)
  log "Download pretrained model and test-data from $repo_url"
  pushd $dir
  GIT_LFS_SKIP_SMUDGE=1 git clone $repo_url
  cd $repo
  git lfs pull --include "*.onnx"
  ls -lh *.onnx
  popd

  python3 ./python-api-examples/offline-decode-files.py \
    --tokens=$repo/tokens.txt \
    --wenet-ctc=$repo/model.onnx \
    $repo/test_wavs/0.wav \
    $repo/test_wavs/1.wav \
    $repo/test_wavs/8k.wav

  python3 ./python-api-examples/online-decode-files.py \
    --tokens=$repo/tokens.txt \
    --wenet-ctc=$repo/model-streaming.onnx \
    $repo/test_wavs/0.wav \
    $repo/test_wavs/1.wav \
    $repo/test_wavs/8k.wav

  python3 sherpa-onnx/python/tests/test_offline_recognizer.py --verbose

  python3 sherpa-onnx/python/tests/test_online_recognizer.py --verbose

  rm -rf $repo
done

log "Offline TTS test"
# test waves are saved in ./tts
mkdir ./tts

log "vits-ljs test"

wget -qq https://huggingface.co/csukuangfj/vits-ljs/resolve/main/vits-ljs.onnx
wget -qq https://huggingface.co/csukuangfj/vits-ljs/resolve/main/lexicon.txt
wget -qq https://huggingface.co/csukuangfj/vits-ljs/resolve/main/tokens.txt

python3 ./python-api-examples/offline-tts.py \
  --vits-model=./vits-ljs.onnx \
  --vits-lexicon=./lexicon.txt \
  --vits-tokens=./tokens.txt \
  --output-filename=./tts/vits-ljs.wav \
  'liliana, the most beautiful and lovely assistant of our team!'

ls -lh ./tts

rm -v vits-ljs.onnx ./lexicon.txt ./tokens.txt

log "vits-vctk test"
wget -qq https://huggingface.co/csukuangfj/vits-vctk/resolve/main/vits-vctk.onnx
wget -qq https://huggingface.co/csukuangfj/vits-vctk/resolve/main/lexicon.txt
wget -qq https://huggingface.co/csukuangfj/vits-vctk/resolve/main/tokens.txt

for sid in 0 10 90; do
  python3 ./python-api-examples/offline-tts.py \
    --vits-model=./vits-vctk.onnx \
    --vits-lexicon=./lexicon.txt \
    --vits-tokens=./tokens.txt \
    --sid=$sid \
    --output-filename=./tts/vits-vctk-${sid}.wav \
    'liliana, the most beautiful and lovely assistant of our team!'
done

rm -v vits-vctk.onnx ./lexicon.txt ./tokens.txt

log "vits-zh-aishell3"

wget -qq https://huggingface.co/csukuangfj/vits-zh-aishell3/resolve/main/vits-aishell3.onnx
wget -qq https://huggingface.co/csukuangfj/vits-zh-aishell3/resolve/main/lexicon.txt
wget -qq https://huggingface.co/csukuangfj/vits-zh-aishell3/resolve/main/tokens.txt

for sid in 0 10 90; do
  python3 ./python-api-examples/offline-tts.py \
    --vits-model=./vits-aishell3.onnx \
    --vits-lexicon=./lexicon.txt \
    --vits-tokens=./tokens.txt \
    --sid=$sid \
    --output-filename=./tts/vits-aishell3-${sid}.wav \
    '林美丽最美丽'
done

rm -v vits-aishell3.onnx ./lexicon.txt ./tokens.txt

mkdir -p /tmp/icefall-models
dir=/tmp/icefall-models

log "Test streaming transducer models"

pushd $dir
repo_url=https://huggingface.co/csukuangfj/sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20

log "Start testing ${repo_url}"
repo=$dir/$(basename $repo_url)
log "Download pretrained model and test-data from $repo_url"

GIT_LFS_SKIP_SMUDGE=1 git clone $repo_url
cd $repo
git lfs pull --include "*.onnx"
popd

python3 -c "import sherpa_onnx; print(sherpa_onnx.__file__)"
sherpa_onnx_version=$(python3 -c "import sherpa_onnx; print(sherpa_onnx.__version__)")

echo "sherpa_onnx version: $sherpa_onnx_version"

pwd
ls -lh

ls -lh $repo

python3 ./python-api-examples/online-decode-files.py \
  --tokens=$repo/tokens.txt \
  --encoder=$repo/encoder-epoch-99-avg-1.onnx \
  --decoder=$repo/decoder-epoch-99-avg-1.onnx \
  --joiner=$repo/joiner-epoch-99-avg-1.onnx \
  $repo/test_wavs/0.wav \
  $repo/test_wavs/1.wav \
  $repo/test_wavs/2.wav \
  $repo/test_wavs/3.wav \
  $repo/test_wavs/8k.wav

python3 ./python-api-examples/online-decode-files.py \
  --tokens=$repo/tokens.txt \
  --encoder=$repo/encoder-epoch-99-avg-1.int8.onnx \
  --decoder=$repo/decoder-epoch-99-avg-1.int8.onnx \
  --joiner=$repo/joiner-epoch-99-avg-1.int8.onnx \
  $repo/test_wavs/0.wav \
  $repo/test_wavs/1.wav \
  $repo/test_wavs/2.wav \
  $repo/test_wavs/3.wav \
  $repo/test_wavs/8k.wav

python3 sherpa-onnx/python/tests/test_online_recognizer.py --verbose

log "Test non-streaming transducer models"

pushd $dir
repo_url=https://huggingface.co/csukuangfj/sherpa-onnx-zipformer-en-2023-04-01

log "Start testing ${repo_url}"
repo=$dir/$(basename $repo_url)
log "Download pretrained model and test-data from $repo_url"

GIT_LFS_SKIP_SMUDGE=1 git clone $repo_url
cd $repo
git lfs pull --include "*.onnx"
popd

ls -lh $repo

python3 ./python-api-examples/offline-decode-files.py \
  --tokens=$repo/tokens.txt \
  --encoder=$repo/encoder-epoch-99-avg-1.onnx \
  --decoder=$repo/decoder-epoch-99-avg-1.onnx \
  --joiner=$repo/joiner-epoch-99-avg-1.onnx \
  $repo/test_wavs/0.wav \
  $repo/test_wavs/1.wav \
  $repo/test_wavs/8k.wav

python3 ./python-api-examples/offline-decode-files.py \
  --tokens=$repo/tokens.txt \
  --encoder=$repo/encoder-epoch-99-avg-1.int8.onnx \
  --decoder=$repo/decoder-epoch-99-avg-1.int8.onnx \
  --joiner=$repo/joiner-epoch-99-avg-1.int8.onnx \
  $repo/test_wavs/0.wav \
  $repo/test_wavs/1.wav \
  $repo/test_wavs/8k.wav

python3 sherpa-onnx/python/tests/test_offline_recognizer.py --verbose

rm -rf $repo

log "Test non-streaming paraformer models"

pushd $dir
repo_url=https://huggingface.co/csukuangfj/sherpa-onnx-paraformer-zh-2023-03-28

log "Start testing ${repo_url}"
repo=$dir/$(basename $repo_url)
log "Download pretrained model and test-data from $repo_url"

GIT_LFS_SKIP_SMUDGE=1 git clone $repo_url
cd $repo
git lfs pull --include "*.onnx"
popd

ls -lh $repo

python3 ./python-api-examples/offline-decode-files.py \
  --tokens=$repo/tokens.txt \
  --paraformer=$repo/model.onnx \
  $repo/test_wavs/0.wav \
  $repo/test_wavs/1.wav \
  $repo/test_wavs/2.wav \
  $repo/test_wavs/8k.wav

python3 ./python-api-examples/offline-decode-files.py \
  --tokens=$repo/tokens.txt \
  --paraformer=$repo/model.int8.onnx \
  $repo/test_wavs/0.wav \
  $repo/test_wavs/1.wav \
  $repo/test_wavs/2.wav \
  $repo/test_wavs/8k.wav

python3 sherpa-onnx/python/tests/test_offline_recognizer.py --verbose

rm -rf $repo

log "Test non-streaming NeMo CTC models"

pushd $dir
repo_url=http://huggingface.co/csukuangfj/sherpa-onnx-nemo-ctc-en-citrinet-512

log "Start testing ${repo_url}"
repo=$dir/$(basename $repo_url)
log "Download pretrained model and test-data from $repo_url"

GIT_LFS_SKIP_SMUDGE=1 git clone $repo_url
cd $repo
git lfs pull --include "*.onnx"
popd

ls -lh $repo

python3 ./python-api-examples/offline-decode-files.py \
  --tokens=$repo/tokens.txt \
  --nemo-ctc=$repo/model.onnx \
  $repo/test_wavs/0.wav \
  $repo/test_wavs/1.wav \
  $repo/test_wavs/8k.wav

python3 ./python-api-examples/offline-decode-files.py \
  --tokens=$repo/tokens.txt \
  --nemo-ctc=$repo/model.int8.onnx \
  $repo/test_wavs/0.wav \
  $repo/test_wavs/1.wav \
  $repo/test_wavs/8k.wav

python3 sherpa-onnx/python/tests/test_offline_recognizer.py --verbose

rm -rf $repo

# test text2token
git clone https://github.com/pkufool/sherpa-test-data /tmp/sherpa-test-data

python3 sherpa-onnx/python/tests/test_text2token.py --verbose

rm -rf /tmp/sherpa-test-data
