#!/usr/bin/env bash

set -ex

curl -SL -O https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20.tar.bz2
tar xvf sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20.tar.bz2
rm sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20.tar.bz2

src=sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20
dst=$src-mobile

mkdir -p $dst

./run-impl.sh \
  --input $src/encoder-epoch-99-avg-1.onnx \
  --output1 $dst/encoder-epoch-99-avg-1.onnx \
  --output2 $dst/encoder-epoch-99-avg-1.int8.onnx

cp -v $src/README.md $dst/
cp -v $src/tokens.txt $dst/
cp -av $src/test_wavs $dst/
cp -v $src/decoder-epoch-99-avg-1.onnx $dst/
cp -v $src/joiner-epoch-99-avg-1.int8.onnx $dst/

cat > $dst/notes.md <<EOF
# Introduction
This model is converted from
https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/$src.tar.bz2
and it supports only batch size equal to 1.
EOF

echo "---$src---"
ls -lh $src
echo "---$dst---"
ls -lh $dst
rm -rf $src

tar cjfv $dst.tar.bz2 $dst
mv *.tar.bz2 ../../
rm -rf $dst
