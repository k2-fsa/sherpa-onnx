#!/usr/bin/env bash
#
# Test group: streaming ASR, decoding graphs, language models and keyword
# spotting.
#
# One of several groups run in parallel by .github/workflows/run-python-test.yaml.
# Run .github/scripts/test-python.sh to run every group sequentially.

set -ex

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/common.sh"

export GIT_CLONE_PROTECTION_ACTIVE=false

log "test T-one"

download_and_extract https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-streaming-t-one-russian-2025-09-08.tar.bz2

python3 ./python-api-examples/online-t-one-ctc-decode-files.py

rm -rf sherpa-onnx-streaming-t-one-russian-2025-09-08

log "test online NeMo CTC"

url=https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-nemo-streaming-fast-conformer-ctc-en-80ms.tar.bz2
name=$(basename $url)
repo=$(basename -s .tar.bz2 $name)

download_and_extract $url
ls -lh $repo
python3 ./python-api-examples/online-nemo-ctc-decode-files.py
rm -rf $repo

log "test streaming zipformer2 ctc HLG decoding"

download_and_extract https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-streaming-zipformer-ctc-small-2024-03-18.tar.bz2
repo=sherpa-onnx-streaming-zipformer-ctc-small-2024-03-18

python3 ./python-api-examples/online-zipformer-ctc-hlg-decode-file.py \
  --debug 1 \
  --tokens ./sherpa-onnx-streaming-zipformer-ctc-small-2024-03-18/tokens.txt \
  --graph ./sherpa-onnx-streaming-zipformer-ctc-small-2024-03-18/HLG.fst \
  --model ./sherpa-onnx-streaming-zipformer-ctc-small-2024-03-18/ctc-epoch-30-avg-3-chunk-16-left-128.int8.onnx \
  ./sherpa-onnx-streaming-zipformer-ctc-small-2024-03-18/test_wavs/0.wav

rm -rf sherpa-onnx-streaming-zipformer-ctc-small-2024-03-18

log "test offline zipformer ctc HLG decoding"

download_and_extract https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-zipformer-ctc-en-2023-10-02.tar.bz2

python3 ./python-api-examples/offline-zipformer-ctc-hlg-decode-files.py \
  --tokens ./sherpa-onnx-zipformer-ctc-en-2023-10-02/tokens.txt \
  --model ./sherpa-onnx-zipformer-ctc-en-2023-10-02/model.int8.onnx \
  --graph ./sherpa-onnx-zipformer-ctc-en-2023-10-02/HLG.fst \
  ./sherpa-onnx-zipformer-ctc-en-2023-10-02/test_wavs/0.wav

rm -rf sherpa-onnx-zipformer-ctc-en-2023-10-02

mkdir -p /tmp/icefall-models
dir=/tmp/icefall-models

pushd $dir

repo=$dir/icefall-asr-librispeech-streaming-zipformer-small-2024-03-18
mkdir -p $repo
cd $repo
mkdir exp-ctc-rnnt-small
cd exp-ctc-rnnt-small
download https://huggingface.co/csukuangfj/icefall-asr-librispeech-streaming-zipformer-small-2024-03-18/resolve/main/exp-ctc-rnnt-small/ctc-epoch-30-avg-3-chunk-16-left-128.int8.onnx
cd ..
mkdir -p data/lang_bpe_500
cd data/lang_bpe_500
download https://huggingface.co/csukuangfj/icefall-asr-librispeech-streaming-zipformer-small-2024-03-18/resolve/main/data/lang_bpe_500/tokens.txt
cd ../..
mkdir test_wavs
cd test_wavs

download https://huggingface.co/csukuangfj/icefall-asr-librispeech-streaming-zipformer-small-2024-03-18/resolve/main/test_wavs/0.wav
download https://huggingface.co/csukuangfj/icefall-asr-librispeech-streaming-zipformer-small-2024-03-18/resolve/main/test_wavs/1.wav
download https://huggingface.co/csukuangfj/icefall-asr-librispeech-streaming-zipformer-small-2024-03-18/resolve/main/test_wavs/8k.wav
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
# sherpa-onnx-en-wenet-gigaspeech
)

for name in ${wenet_models[@]}; do
  repo_url=https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/$name.tar.bz2
  download_and_extract $repo_url
  repo=$name
  log "Start testing ${repo_url}"

  if false; then
    # offline wenet ctc models are not supported by onnxruntime >= 1.18
    python3 ./python-api-examples/offline-decode-files.py \
      --tokens=$repo/tokens.txt \
      --wenet-ctc=$repo/model.onnx \
      $repo/test_wavs/0.wav \
      $repo/test_wavs/1.wav \
      $repo/test_wavs/8k.wav
  fi

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

mkdir -p /tmp/icefall-models
dir=/tmp/icefall-models

log "Test streaming transducer models"

if [[ x$OS != x'windows-latest' ]]; then
  echo "OS: $OS"
  pushd $dir
  repo_url=https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20.tar.bz2
  download_and_extract $repo_url
  repo=sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20

  log "Start testing ${repo_url}"
  repo=$dir/$repo

  python3 -c "import sherpa_onnx; print(sherpa_onnx.__file__)"
  sherpa_onnx_version=$(python3 -c "import sherpa_onnx; print(sherpa_onnx.__version__)")

  echo "sherpa_onnx version: $sherpa_onnx_version"

  pwd
  ls -lh

  ls -lh $repo
  popd

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
    --decoder=$repo/decoder-epoch-99-avg-1.onnx \
    --joiner=$repo/joiner-epoch-99-avg-1.int8.onnx \
    $repo/test_wavs/0.wav \
    $repo/test_wavs/1.wav \
    $repo/test_wavs/2.wav \
    $repo/test_wavs/3.wav \
    $repo/test_wavs/8k.wav

  ln -s $repo $PWD/

  download https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/itn_zh_number.fst
  download https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/itn-zh-number.wav

  python3 ./python-api-examples/inverse-text-normalization-online-asr.py

  python3 sherpa-onnx/python/tests/test_online_recognizer.py --verbose

  rm -rfv sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20

  rm -rf $repo
fi

log "Test non-streaming transducer models"

pushd $dir
repo_url=https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-zipformer-en-2023-04-01.tar.bz2
log "Download pretrained model and test-data from $repo_url"

download_and_extract $repo_url
repo=$dir/sherpa-onnx-zipformer-en-2023-04-01

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
  --decoder=$repo/decoder-epoch-99-avg-1.onnx \
  --joiner=$repo/joiner-epoch-99-avg-1.int8.onnx \
  $repo/test_wavs/0.wav \
  $repo/test_wavs/1.wav \
  $repo/test_wavs/8k.wav

lm_repo_url=https://huggingface.co/ezerhouni/icefall-librispeech-rnn-lm
log "Download pre-trained RNN-LM model from ${lm_repo_url}"
GIT_LFS_SKIP_SMUDGE=1 git clone $lm_repo_url
lm_repo=$(basename $lm_repo_url)
pushd $lm_repo
git lfs pull --include "exp/no-state-epoch-99-avg-1.onnx"
popd

bigram_repo_url=https://huggingface.co/vsd-vector/librispeech_bigram_sherpa-onnx-zipformer-large-en-2023-06-26
log "Download bi-gram LM from ${bigram_repo_url}"
GIT_LFS_SKIP_SMUDGE=1 git clone $bigram_repo_url
bigramlm_repo=$(basename $bigram_repo_url)
pushd $bigramlm_repo
git lfs pull --include "2gram.fst"
popd

log "Perform offline decoding with RNN-LM and LODR"
python3 ./python-api-examples/offline-decode-files.py \
  --tokens=$repo/tokens.txt \
  --encoder=$repo/encoder-epoch-99-avg-1.onnx \
  --decoder=$repo/decoder-epoch-99-avg-1.onnx \
  --joiner=$repo/joiner-epoch-99-avg-1.onnx \
  --decoding-method=modified_beam_search \
  --lm=$lm_repo/exp/no-state-epoch-99-avg-1.onnx \
  --lodr-fst=$bigramlm_repo/2gram.fst \
  --lodr-scale=-0.5 \
  $repo/test_wavs/0.wav \
  $repo/test_wavs/1.wav \
  $repo/test_wavs/8k.wav

python3 sherpa-onnx/python/tests/test_offline_recognizer.py --verbose

rm -rf $repo $lm_repo $bigramlm_repo

log "Test non-streaming paraformer models"

if [[ x$OS != x'windows-latest' ]]; then
  echo "OS: $OS"
  pushd $dir
  repo_url=https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-paraformer-zh-2023-09-14.tar.bz2
  download_and_extract $repo_url

  log "Start testing ${repo_url}"
  repo=$dir/sherpa-onnx-paraformer-zh-2023-09-14

  ls -lh $repo
  popd

  python3 ./python-api-examples/offline-decode-files.py \
    --tokens=$repo/tokens.txt \
    --paraformer=$repo/model.int8.onnx \
    $repo/test_wavs/0.wav \
    $repo/test_wavs/1.wav \
    $repo/test_wavs/2.wav \
    $repo/test_wavs/8k.wav

  python3 sherpa-onnx/python/tests/test_offline_recognizer.py --verbose

  ln -s $repo $PWD/

  download https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/itn_zh_number.fst
  download https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/itn-zh-number.wav

  python3 ./python-api-examples/inverse-text-normalization-offline-asr.py

  rm -rfv sherpa-onnx-paraformer-zh-2023-09-14

  rm -rf $repo
fi

log "Test non-streaming NeMo CTC models"

pushd $dir
repo_url=https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-nemo-ctc-en-citrinet-512.tar.bz2
download_and_extract $repo_url

log "Start testing ${repo_url}"
repo=$dir/sherpa-onnx-nemo-ctc-en-citrinet-512

ls -lh $repo
popd

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

if [[ x$OS != x'windows-latest' ]]; then
  echo "OS: $OS"

  repo=sherpa-onnx-kws-zipformer-wenetspeech-3.3M-2024-01-01
  log "Start testing ${repo}"

  download_and_extract https://github.com/pkufool/keyword-spotting-models/releases/download/v0.1/sherpa-onnx-kws-zipformer-wenetspeech-3.3M-2024-01-01.tar.bz

  ls -lh $repo

  python3 ./python-api-examples/keyword-spotter.py

  python3 sherpa-onnx/python/tests/test_keyword_spotter.py --verbose

  rm -rf $repo
fi

rm -r $dir

log "test-python-asr-streaming.sh passed"
