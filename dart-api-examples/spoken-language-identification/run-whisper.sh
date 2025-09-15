#!/usr/bin/env bash

set -ex

dart pub get

if [ ! -f ./sherpa-onnx-whisper-tiny/tiny-encoder.int8.onnx ]; then
  curl -SL -O https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-whisper-tiny.tar.bz2
  tar xvf sherpa-onnx-whisper-tiny.tar.bz2
  rm sherpa-onnx-whisper-tiny.tar.bz2
fi

# Download test WAV files
waves=(
# ar-arabic.wav
# bg-bulgarian.wav
# cs-czech.wav
# da-danish.wav
# de-german.wav
# el-greek.wav
en-english.wav
es-spanish.wav
# fa-persian.wav
# fi-finnish.wav
# fr-french.wav
# hi-hindi.wav
# hr-croatian.wav
# id-indonesian.wav
# it-italian.wav
# ja-japanese.wav
# ko-korean.wav
# nl-dutch.wav
# no-norwegian.wav
# pl-polish.wav
# pt-portuguese.wav
# ro-romanian.wav
ru-russian.wav
# sk-slovak.wav
# sv-swedish.wav
# ta-tamil.wav
# tl-tagalog.wav
# tr-turkish.wav
# uk-ukrainian.wav
zh-chinese.wav
)

for wav in ${waves[@]}; do
  if [ ! -f ./$wav ]; then
    echo "Downloading $wav"
    curl -SL -O https://hf-mirror.com/spaces/k2-fsa/spoken-language-identification/resolve/main/test_wavs/$wav
  fi
  
  echo "Testing $wav"
  dart run \
    ./bin/spoken_language_identification.dart \
    --encoder ./sherpa-onnx-whisper-tiny/tiny-encoder.int8.onnx \
    --decoder ./sherpa-onnx-whisper-tiny/tiny-decoder.int8.onnx \
    --wav ./$wav
  
  echo "----------------------------------------"
done
