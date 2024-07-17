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
