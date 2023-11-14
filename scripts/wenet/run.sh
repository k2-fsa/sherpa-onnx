#!/usr/bin/env bash
#
# Copyright      2023  Xiaomi Corp.        (authors: Fangjun Kuang)
#
# Please refer to
# https://github.com/wenet-e2e/wenet/blob/main/docs/pretrained_models.en.md
# for a table of pre-trained models.
# Please select the column "Checkpoint Model" for downloading.

function install_dependencies() {
  pip install torch==2.1.0+cpu torchaudio==2.1.0+cpu -f https://download.pytorch.org/whl/torch_stable.html
  pip install k2==1.24.4.dev20231022+cpu.torch2.1.0 -f https://k2-fsa.github.io/k2/cpu.html

  pip install onnxruntime onnx kaldi-native-fbank pyyaml

  pip install git+https://github.com/wenet-e2e/wenet.git
  wenet_dir=$(dirname $(python3 -c "import wenet; print(wenet.__file__)"))
  git clone https://github.com/wenet-e2e/wenet
  if [ ! -d $wenet_dir/transducer/search ]; then
    cp -av ./wenet/wenet/transducer/search $wenet_dir/transducer
  fi

  if [ ! -d $wenet_dir/e_branchformer ]; then
    cp -a .//wenet/wenet/e_branchformer $wenet_dir
  fi

  if [ ! -d $wenet_dir/ctl_model ]; then
    cp -a .//wenet/wenet/ctl_model $wenet_dir
  fi
}

function aishell() {
  wget -q https://huggingface.co/openspeech/wenet-models/resolve/main/aishell_u2pp_conformer_exp.tar.gz
  tar xvf aishell_u2pp_conformer_exp.tar.gz
  rm -v aishell_u2pp_conformer_exp.tar.gz

  pushd aishell_u2pp_conformer_exp
  mkdir -p exp/20210601_u2++_conformer_exp
  cp global_cmvn ./exp/20210601_u2++_conformer_exp
  cp ../*.py .

  export WENET_URL=https://huggingface.co/openspeech/wenet-models/resolve/main/aishell_u2pp_conformer_exp.tar.gz
  wget -O 0.wav https://huggingface.co/openspeech/wenet-models/resolve/main/zh.wav

  echo "Test streaming"
  ./export-onnx-streaming.py
  ./test-onnx-streaming.py

  echo "Test non-streaming"
  ./export-onnx.py
  ./test-onnx.py
  ls -lh
  popd
}

install_dependencies()

aishell()
