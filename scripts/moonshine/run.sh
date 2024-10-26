#!/usr/bin/env bash
# Copyright      2024  Xiaomi Corp.        (authors: Fangjun Kuang)
set -ex

cat >LICENSE <<EOF
MIT License

Copyright (c) 2024 Useful Sensors

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
EOF

function download_files() {
  for d in tiny base; do
    mkdir $d

    pushd $d
      curl -SL -O https://huggingface.co/UsefulSensors/moonshine/resolve/main/onnx/$d/preprocess.onnx
      curl -SL -O https://huggingface.co/UsefulSensors/moonshine/resolve/main/onnx/$d/encode.onnx
      curl -SL -O https://huggingface.co/UsefulSensors/moonshine/resolve/main/onnx/$d/uncached_decode.onnx
      curl -SL -O https://huggingface.co/UsefulSensors/moonshine/resolve/main/onnx/$d/cached_decode.onnx
    popd
  done

  curl -SL -O https://huggingface.co/csukuangfj/sherpa-onnx-whisper-base/resolve/main/test_wavs/0.wav
  curl -SL -O https://huggingface.co/csukuangfj/sherpa-onnx-whisper-base/resolve/main/test_wavs/1.wav
  curl -SL -O https://huggingface.co/csukuangfj/sherpa-onnx-whisper-base/resolve/main/test_wavs/8k.wav
  curl -SL -O https://huggingface.co/csukuangfj/sherpa-onnx-whisper-base/resolve/main/test_wavs/trans.txt

  curl -SL -O https://raw.githubusercontent.com/usefulsensors/moonshine/refs/heads/main/moonshine/assets/tokenizer.json
}

function quantize() {
  for d in tiny base; do
    echo "==========$d=========="
    ls -lh
    mv $d/*.onnx .
    ./export-onnx.py
    rm cached_decode.onnx
    rm uncached_decode.onnx
    rm encode.onnx
    ls -lh

    ./test.py

    mv *.onnx $d
    mv tokens.txt $d
    ls -lh $d

  done
}

function zip() {
  for d in tiny base; do
    s=sherpa-onnx-moonshine-$d-en-int8
    mv $d $s

    mkdir $s/test_wavs

    cp -v *.wav $s/test_wavs
    cp trans.txt $s/test_wavs
    cp LICENSE $s/
    cp ./README.md $s

    ls -lh $s
    tar cjfv $s.tar.bz2 $s
  done
}

download_files
quantize
zip

ls -lh
