#!/usr/bin/env bash

set -ex


function install() {
  pip install torch==2.3.1+cpu torchaudio==2.3.1+cpu -f https://download.pytorch.org/whl/torch_stable.html

  pushd /tmp

  git clone https://github.com/alibaba/FunASR.git
  cd FunASR
  pip3 install -qq -e ./
  cd ..

  git clone https://github.com/FunAudioLLM/SenseVoice
  cd SenseVoice
  pip install -qq -r ./requirements.txt
  cd ..

  pip install soundfile onnx onnxruntime kaldi-native-fbank librosa soundfile

  popd
}

install

export PYTHONPATH=/tmp/FunASR:$PYTHONPATH
export PYTHONPATH=/tmp/SenseVoice:$PYTHONPATH

echo "pwd: $PWD"

./export-onnx.py

./show-info.py

ls -lh

# Download test wavs
curl -SL -O https://huggingface.co/csukuangfj/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17/resolve/main/test_wavs/en.wav
curl -SL -O https://huggingface.co/csukuangfj/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17/resolve/main/test_wavs/zh.wav
curl -SL -O https://huggingface.co/csukuangfj/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17/resolve/main/test_wavs/ja.wav
curl -SL -O https://huggingface.co/csukuangfj/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17/resolve/main/test_wavs/ko.wav
curl -SL -O https://huggingface.co/csukuangfj/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17/resolve/main/test_wavs/yue.wav

for m in model.onnx model.int8.onnx; do
  for w in en zh ja ko yue; do
    echo "----------test $m $w.wav----------"

    echo "without inverse text normalization, lang auto"
    ./test.py --model $m --tokens ./tokens.txt --wav $w.wav --use-itn 0

    echo "with inverse text normalization, lang auto"
    ./test.py --model $m --tokens ./tokens.txt --wav $w.wav --use-itn 1

    echo "without inverse text normalization, lang $w"
    ./test.py --model $m --tokens ./tokens.txt --wav $w.wav --use-itn 0 --lang $w

    echo "with inverse text normalization, lang $w"
    ./test.py --model $m --tokens ./tokens.txt --wav $w.wav --use-itn 1 --lang $w
  done
done
