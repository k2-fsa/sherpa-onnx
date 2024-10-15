#!/usr/bin/env bash

set -ex

cur_dir=$(cd $(dirname $BASH_SOURCE) && pwd)
sherpa_onnx_dir=$(cd $cur_dir/../.. && pwd)
echo "sherpa_onnx_dir: $sherpa_onnx_dir"

pip install sherpa-onnx # for testing

function download_model() {
  git lfs install
  git clone https://www.modelscope.cn/pkufool/icefall-asr-zipformer-libriheavy-20230926.git
}

function download_test_wavs() {
  d=$1
  mkdir $d/test_wavs
  pushd $d/test_wavs
  curl -SL -O https://huggingface.co/csukuangfj/sherpa-onnx-whisper-medium.en/resolve/main/test_wavs/0.wav
  curl -SL -O https://huggingface.co/csukuangfj/sherpa-onnx-whisper-medium.en/resolve/main/test_wavs/1.wav
  curl -SL -O https://huggingface.co/csukuangfj/sherpa-onnx-whisper-medium.en/resolve/main/test_wavs/8k.wav
  curl -SL -O https://huggingface.co/csukuangfj/sherpa-onnx-whisper-medium.en/resolve/main/test_wavs/trans.txt
  popd
}

function export_large() {
  echo "----------large----------"
  src=icefall-asr-zipformer-libriheavy-20230926
  dst=sherpa-onnx-zipformer-en-libriheavy-20230926-large
  mkdir $dst

  cp -v $src/data/lang_bpe_500/bpe.model $dst/
  cp -v $src/data/lang_bpe_500/tokens.txt $dst/
  cp -v $src/exp/*.onnx $dst/
  download_test_wavs $dst

  ls -lh $dst
  ls -lh $dst/test_wavs

  sherpa-onnx-offline \
    --encoder=$dst/encoder-epoch-16-avg-3.onnx \
    --decoder=$dst/decoder-epoch-16-avg-3.onnx \
    --joiner=$dst/joiner-epoch-16-avg-3.onnx \
    --tokens=$dst/tokens.txt \
    $dst/test_wavs/0.wav \
    $dst/test_wavs/1.wav \
    $dst/test_wavs/8k.wav

  sherpa-onnx-offline \
    --encoder=$dst/encoder-epoch-16-avg-3.int8.onnx \
    --decoder=$dst/decoder-epoch-16-avg-3.onnx \
    --joiner=$dst/joiner-epoch-16-avg-3.int8.onnx \
    --tokens=$dst/tokens.txt \
    $dst/test_wavs/0.wav \
    $dst/test_wavs/1.wav \
    $dst/test_wavs/8k.wav
}

function export_medium() {
  echo "----------medium subset----------"
  src=icefall-asr-zipformer-libriheavy-20230926
  dst=sherpa-onnx-zipformer-en-libriheavy-20230926-medium
  mkdir $dst

  cp -v $src/data/lang_bpe_500/bpe.model $dst/
  cp -v $src/data/lang_bpe_500/tokens.txt $dst/
  cp -v $src/exp_medium_subset/*.onnx $dst/
  download_test_wavs $dst

  ls -lh $dst
  ls -lh $dst/test_wavs

  sherpa-onnx-offline \
    --encoder=$dst/encoder-epoch-60-avg-20.onnx \
    --decoder=$dst/decoder-epoch-60-avg-20.onnx \
    --joiner=$dst/joiner-epoch-60-avg-20.onnx \
    --tokens=$dst/tokens.txt \
    $dst/test_wavs/0.wav \
    $dst/test_wavs/1.wav \
    $dst/test_wavs/8k.wav

  sherpa-onnx-offline \
    --encoder=$dst/encoder-epoch-60-avg-20.int8.onnx \
    --decoder=$dst/decoder-epoch-60-avg-20.onnx \
    --joiner=$dst/joiner-epoch-60-avg-20.int8.onnx \
    --tokens=$dst/tokens.txt \
    $dst/test_wavs/0.wav \
    $dst/test_wavs/1.wav \
    $dst/test_wavs/8k.wav
}

function export_small() {
  echo "----------small subset----------"
  src=icefall-asr-zipformer-libriheavy-20230926
  dst=sherpa-onnx-zipformer-en-libriheavy-20230926-small
  mkdir $dst

  cp -v $src/data/lang_bpe_500/bpe.model $dst/
  cp -v $src/data/lang_bpe_500/tokens.txt $dst/
  cp -v $src/exp_small_subset/*.onnx $dst/
  download_test_wavs $dst

  ls -lh $dst
  ls -lh $dst/test_wavs

  sherpa-onnx-offline \
    --encoder=$dst/encoder-epoch-90-avg-20.onnx \
    --decoder=$dst/decoder-epoch-90-avg-20.onnx \
    --joiner=$dst/joiner-epoch-90-avg-20.onnx \
    --tokens=$dst/tokens.txt \
    $dst/test_wavs/0.wav \
    $dst/test_wavs/1.wav \
    $dst/test_wavs/8k.wav

  sherpa-onnx-offline \
    --encoder=$dst/encoder-epoch-90-avg-20.int8.onnx \
    --decoder=$dst/decoder-epoch-90-avg-20.onnx \
    --joiner=$dst/joiner-epoch-90-avg-20.int8.onnx \
    --tokens=$dst/tokens.txt \
    $dst/test_wavs/0.wav \
    $dst/test_wavs/1.wav \
    $dst/test_wavs/8k.wav
}

download_model

export_large
export_medium
export_small

rm -rf icefall-asr-zipformer-libriheavy-20230926
