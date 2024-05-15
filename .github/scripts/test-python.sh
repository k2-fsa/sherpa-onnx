#!/usr/bin/env bash

set -ex

log() {
  # This function is from espnet
  local fname=${BASH_SOURCE[1]##*/}
  echo -e "$(date '+%Y-%m-%d %H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}

log "test online NeMo CTC"

url=https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-nemo-streaming-fast-conformer-ctc-en-80ms.tar.bz2
name=$(basename $url)
repo=$(basename -s .tar.bz2 $name)

curl -SL -O $url
tar xvf $name
rm $name
ls -lh $repo
python3 ./python-api-examples/online-nemo-ctc-decode-files.py
rm -rf $repo

log "test offline punctuation"

curl -SL -O https://github.com/k2-fsa/sherpa-onnx/releases/download/punctuation-models/sherpa-onnx-punct-ct-transformer-zh-en-vocab272727-2024-04-12.tar.bz2
tar xvf sherpa-onnx-punct-ct-transformer-zh-en-vocab272727-2024-04-12.tar.bz2
rm sherpa-onnx-punct-ct-transformer-zh-en-vocab272727-2024-04-12.tar.bz2
repo=sherpa-onnx-punct-ct-transformer-zh-en-vocab272727-2024-04-12
ls -lh $repo

python3 ./python-api-examples/add-punctuation.py

rm -rf $repo

log "test audio tagging"

curl -SL -O https://github.com/k2-fsa/sherpa-onnx/releases/download/audio-tagging-models/sherpa-onnx-zipformer-audio-tagging-2024-04-09.tar.bz2
tar xvf sherpa-onnx-zipformer-audio-tagging-2024-04-09.tar.bz2
rm sherpa-onnx-zipformer-audio-tagging-2024-04-09.tar.bz2
 python3 ./python-api-examples/audio-tagging-from-a-file.py
rm -rf sherpa-onnx-zipformer-audio-tagging-2024-04-09


log "test streaming zipformer2 ctc HLG decoding"

curl -SL -O https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-streaming-zipformer-ctc-small-2024-03-18.tar.bz2
tar xvf sherpa-onnx-streaming-zipformer-ctc-small-2024-03-18.tar.bz2
rm sherpa-onnx-streaming-zipformer-ctc-small-2024-03-18.tar.bz2
repo=sherpa-onnx-streaming-zipformer-ctc-small-2024-03-18

python3 ./python-api-examples/online-zipformer-ctc-hlg-decode-file.py \
  --debug 1 \
  --tokens ./sherpa-onnx-streaming-zipformer-ctc-small-2024-03-18/tokens.txt \
  --graph ./sherpa-onnx-streaming-zipformer-ctc-small-2024-03-18/HLG.fst \
  --model ./sherpa-onnx-streaming-zipformer-ctc-small-2024-03-18/ctc-epoch-30-avg-3-chunk-16-left-128.int8.onnx \
  ./sherpa-onnx-streaming-zipformer-ctc-small-2024-03-18/test_wavs/0.wav

rm -rf sherpa-onnx-streaming-zipformer-ctc-small-2024-03-18


mkdir -p /tmp/icefall-models
dir=/tmp/icefall-models

pushd $dir

repo=$dir/icefall-asr-librispeech-streaming-zipformer-small-2024-03-18
mkdir -p $repo
cd $repo
mkdir exp-ctc-rnnt-small
cd exp-ctc-rnnt-small
curl -LS -O https://huggingface.co/csukuangfj/icefall-asr-librispeech-streaming-zipformer-small-2024-03-18/resolve/main/exp-ctc-rnnt-small/ctc-epoch-30-avg-3-chunk-16-left-128.int8.onnx
cd ..
mkdir -p data/lang_bpe_500
cd data/lang_bpe_500
curl -LS -O https://huggingface.co/csukuangfj/icefall-asr-librispeech-streaming-zipformer-small-2024-03-18/resolve/main/data/lang_bpe_500/tokens.txt
cd ../..
mkdir test_wavs
cd test_wavs

curl -LS -O https://huggingface.co/csukuangfj/icefall-asr-librispeech-streaming-zipformer-small-2024-03-18/resolve/main/test_wavs/0.wav
curl -LS -O https://huggingface.co/csukuangfj/icefall-asr-librispeech-streaming-zipformer-small-2024-03-18/resolve/main/test_wavs/1.wav
curl -LS -O https://huggingface.co/csukuangfj/icefall-asr-librispeech-streaming-zipformer-small-2024-03-18/resolve/main/test_wavs/8k.wav
popd

python3 ./python-api-examples/online-decode-files.py \
  --tokens=$repo/data/lang_bpe_500/tokens.txt \
  --zipformer2-ctc=$repo/exp-ctc-rnnt-small/ctc-epoch-30-avg-3-chunk-16-left-128.int8.onnx \
  $repo/test_wavs/0.wav \
  $repo/test_wavs/1.wav \
  $repo/test_wavs/8k.wav

rm -rf $repo

python3 sherpa-onnx/python/tests/test_offline_recognizer.py --verbose

wenet_models=(
# sherpa-onnx-zh-wenet-aishell
# sherpa-onnx-zh-wenet-aishell2
# sherpa-onnx-zh-wenet-wenetspeech
# sherpa-onnx-zh-wenet-multi-cn
sherpa-onnx-en-wenet-librispeech
sherpa-onnx-en-wenet-gigaspeech
)

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

curl -LS -O https://huggingface.co/csukuangfj/vits-ljs/resolve/main/vits-ljs.onnx
curl -LS -O https://huggingface.co/csukuangfj/vits-ljs/resolve/main/lexicon.txt
curl -LS -O https://huggingface.co/csukuangfj/vits-ljs/resolve/main/tokens.txt

python3 ./python-api-examples/offline-tts.py \
  --vits-model=./vits-ljs.onnx \
  --vits-lexicon=./lexicon.txt \
  --vits-tokens=./tokens.txt \
  --output-filename=./tts/vits-ljs.wav \
  'liliana, the most beautiful and lovely assistant of our team!'

ls -lh ./tts

rm -v vits-ljs.onnx ./lexicon.txt ./tokens.txt

log "vits-vctk test"
curl -LS -O https://huggingface.co/csukuangfj/vits-vctk/resolve/main/vits-vctk.onnx
curl -LS -O https://huggingface.co/csukuangfj/vits-vctk/resolve/main/lexicon.txt
curl -LS -O https://huggingface.co/csukuangfj/vits-vctk/resolve/main/tokens.txt

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

if [[ x$OS != x'windows-latest' ]]; then
  echo "OS: $OS"

  log "vits-zh-aishell3"

  curl -LS -O https://huggingface.co/csukuangfj/vits-zh-aishell3/resolve/main/vits-aishell3.onnx
  curl -LS -O https://huggingface.co/csukuangfj/vits-zh-aishell3/resolve/main/lexicon.txt
  curl -LS -O https://huggingface.co/csukuangfj/vits-zh-aishell3/resolve/main/tokens.txt

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
fi

mkdir -p /tmp/icefall-models
dir=/tmp/icefall-models

log "Test streaming transducer models"

if [[ x$OS != x'windows-latest' ]]; then
  echo "OS: $OS"
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
fi

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

if [[ x$OS != x'windows-latest' ]]; then
  echo "OS: $OS"
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
fi

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

dir=/tmp/onnx-models
mkdir -p $dir

log "Test keyword spotting models"

python3 -c "import sherpa_onnx; print(sherpa_onnx.__file__)"
sherpa_onnx_version=$(python3 -c "import sherpa_onnx; print(sherpa_onnx.__version__)")

echo "sherpa_onnx version: $sherpa_onnx_version"

pwd
ls -lh

repo=sherpa-onnx-kws-zipformer-gigaspeech-3.3M-2024-01-01
log "Start testing ${repo}"

pushd $dir
curl -LS -O https://github.com/pkufool/keyword-spotting-models/releases/download/v0.1/sherpa-onnx-kws-zipformer-gigaspeech-3.3M-2024-01-01.tar.bz
tar xf sherpa-onnx-kws-zipformer-gigaspeech-3.3M-2024-01-01.tar.bz
popd

repo=$dir/$repo
ls -lh $repo

python3 ./python-api-examples/keyword-spotter.py \
  --tokens=$repo/tokens.txt \
  --encoder=$repo/encoder-epoch-12-avg-2-chunk-16-left-64.onnx \
  --decoder=$repo/decoder-epoch-12-avg-2-chunk-16-left-64.onnx \
  --joiner=$repo/joiner-epoch-12-avg-2-chunk-16-left-64.onnx \
  --keywords-file=$repo/test_wavs/test_keywords.txt \
  $repo/test_wavs/0.wav \
  $repo/test_wavs/1.wav

rm -rf $repo

if [[ x$OS != x'windows-latest' ]]; then
  echo "OS: $OS"

  repo=sherpa-onnx-kws-zipformer-wenetspeech-3.3M-2024-01-01
  log "Start testing ${repo}"

  pushd $dir
  curl -LS -O https://github.com/pkufool/keyword-spotting-models/releases/download/v0.1/sherpa-onnx-kws-zipformer-wenetspeech-3.3M-2024-01-01.tar.bz
  tar xf sherpa-onnx-kws-zipformer-wenetspeech-3.3M-2024-01-01.tar.bz
  popd

  repo=$dir/$repo
  ls -lh $repo

  python3 ./python-api-examples/keyword-spotter.py \
    --tokens=$repo/tokens.txt \
    --encoder=$repo/encoder-epoch-12-avg-2-chunk-16-left-64.onnx \
    --decoder=$repo/decoder-epoch-12-avg-2-chunk-16-left-64.onnx \
    --joiner=$repo/joiner-epoch-12-avg-2-chunk-16-left-64.onnx \
    --keywords-file=$repo/test_wavs/test_keywords.txt \
    $repo/test_wavs/3.wav \
    $repo/test_wavs/4.wav \
    $repo/test_wavs/5.wav

  python3 sherpa-onnx/python/tests/test_keyword_spotter.py --verbose

  rm -rf $repo
fi

rm -r $dir
