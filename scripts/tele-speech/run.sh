#!/usr/bin/env bash

curl -SL -O https://hf-mirror.com/lovemefan/telespeech/resolve/main/model_export.onnx

mv model_export.onnx model.onnx

curl -SL -O https://hf-mirror.com/lovemefan/telespeech/resolve/main/vocab.json

curl -SL -O https://github.com/csukuangfj/models/releases/download/a/TeleSpeech.pdf
curl -SL -O https://hf-mirror.com/csukuangfj/sherpa-onnx-paraformer-zh-small-2024-03-09/resolve/main/test_wavs/3-sichuan.wav
curl -SL -O https://hf-mirror.com/csukuangfj/sherpa-onnx-paraformer-zh-small-2024-03-09/resolve/main/test_wavs/4-tianjin.wav
curl -SL -O https://hf-mirror.com/csukuangfj/sherpa-onnx-paraformer-zh-small-2024-03-09/resolve/main/test_wavs/5-henan.wav

ls -lh

./add-metadata.py

dst=sherpa-onnx-telespeech-ctc-zh-2024-06-04
mkdir $dst
mkdir $dst/test_wavs
cp -v model.onnx $dst/
cp -v tokens.txt $dst
cp -v *.wav $dst/test_wavs
cp -v *.pdf $dst
cp -v README.md $dst
cp -v *.py $dst

ls -lh $dst

tar cvjfv ${dst}.tar.bz2 $dst

dst=sherpa-onnx-telespeech-ctc-int8-zh-2024-06-04
mkdir $dst
mkdir $dst/test_wavs
cp -v model.int8.onnx $dst/
cp -v tokens.txt $dst
cp -v *.wav $dst/test_wavs
cp -v *.pdf $dst
cp -v README.md $dst
cp -v *.py $dst

ls -lh $dst

tar cvjfv ${dst}.tar.bz2 $dst

cp -v *.tar.bz2 ../..

ls -lh ../../
