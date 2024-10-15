#!/usr/bin/env bash
#
# Copyright      2023  Xiaomi Corp.        (authors: Fangjun Kuang)
#
# Please refer to
# https://github.com/wenet-e2e/wenet/blob/main/docs/pretrained_models.en.md
# for a table of pre-trained models.
# Please select the column "Checkpoint Model" for downloading.

set -ex

export PYTHONPATH=/tmp/wenet:$PYTHONPATH

function install_dependencies() {
  pip install soundfile
  pip install torch==2.3.1+cpu torchaudio==2.3.1+cpu -f https://download.pytorch.org/whl/torch_stable.html
  pip install k2==1.24.4.dev20240606+cpu.torch2.3.1 -f https://k2-fsa.github.io/k2/cpu.html

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
    cp -a ./wenet/wenet/ctl_model $wenet_dir
  fi

  if [ ! -d $wenet_dir/finetune ]; then
    cp -av ./wenet/wenet/finetune $wenet_dir/
  fi

  mv wenet /tmp
}

function aishell() {
  echo "aishell"
  wget -q https://huggingface.co/openspeech/wenet-models/resolve/main/aishell_u2pp_conformer_exp.tar.gz
  tar xvf aishell_u2pp_conformer_exp.tar.gz
  rm -v aishell_u2pp_conformer_exp.tar.gz

  pushd aishell_u2pp_conformer_exp
  mkdir -p exp/20210601_u2++_conformer_exp
  cp global_cmvn ./exp/20210601_u2++_conformer_exp
  cp ../*.py .

  export WENET_URL=https://wenet.org.cn/downloads?models=wenet&version=aishell_u2pp_conformer_exp.tar.gz
  wget -O 0.wav https://huggingface.co/openspeech/wenet-models/resolve/main/zh.wav
  soxi 0.wav

  echo "Test streaming"
  ./export-onnx-streaming.py
  ls -lh
  ./test-onnx-streaming.py

  echo "Test non-streaming"
  ./export-onnx.py
  ls -lh
  ./test-onnx.py

  cat > README.md <<EOF
# Introduction
This model is converted from https://wenet.org.cn/downloads?models=wenet&version=aishell_u2pp_conformer_exp.tar.gz
EOF

  popd
}

function aishell2() {
  echo "aishell2"
  wget -q https://huggingface.co/openspeech/wenet-models/resolve/main/aishell2_u2pp_conformer_exp.tar.gz
  tar xvf aishell2_u2pp_conformer_exp.tar.gz
  rm -v aishell2_u2pp_conformer_exp.tar.gz

  pushd aishell2_u2pp_conformer_exp
  mkdir -p exp/u2++_conformer
  cp global_cmvn ./exp/u2++_conformer
  cp ../*.py .

  export WENET_URL=https://wenet.org.cn/downloads?models=wenet&version=aishell2_u2pp_conformer_exp.tar.gz
  wget -O 0.wav https://huggingface.co/openspeech/wenet-models/resolve/main/zh.wav
  soxi 0.wav

  echo "Test streaming"
  ./export-onnx-streaming.py
  ls -lh
  ./test-onnx-streaming.py

  echo "Test non-streaming"
  ./export-onnx.py
  ls -lh
  ./test-onnx.py

  cat > README.md <<EOF
# Introduction
This model is converted from https://wenet.org.cn/downloads?models=wenet&version=aishell2_u2pp_conformer_exp.tar.gz
EOF

  popd
}

function multi_cn() {
  echo "multi_cn"
  wget -q https://huggingface.co/openspeech/wenet-models/resolve/main/multi_cn_unified_conformer_exp.tar.gz
  tar xvf multi_cn_unified_conformer_exp.tar.gz
  rm -v multi_cn_unified_conformer_exp.tar.gz

  pushd multi_cn_unified_conformer_exp
  mkdir -p exp/20210815_unified_conformer_exp
  cp global_cmvn ./exp/20210815_unified_conformer_exp
  cp ../*.py .

  export WENET_URL=https://wenet.org.cn/downloads?models=wenet&version=multi_cn_unified_conformer_exp.tar.gz
  wget -O 0.wav https://huggingface.co/openspeech/wenet-models/resolve/main/zh.wav
  soxi 0.wav

  echo "Test streaming"
  ./export-onnx-streaming.py
  ls -lh
  ./test-onnx-streaming.py

  echo "Test non-streaming"
  ./export-onnx.py
  ls -lh
  ./test-onnx.py

  cat > README.md <<EOF
# Introduction
This model is converted from https://wenet.org.cn/downloads?models=wenet&version=multi_cn_unified_conformer_exp.tar.gz
EOF

  popd
}

function wenetspeech() {
  echo "wenetspeech"
  wget -q https://huggingface.co/openspeech/wenet-models/resolve/main/wenetspeech_u2pp_conformer_exp.tar.gz
  tar xvf wenetspeech_u2pp_conformer_exp.tar.gz
  rm -v wenetspeech_u2pp_conformer_exp.tar.gz

  pushd 20220506_u2pp_conformer_exp
  mkdir -p exp/20220506_u2pp_conformer_exp
  cp global_cmvn ./exp/20220506_u2pp_conformer_exp
  cp ../*.py .

  export WENET_URL=https://wenet.org.cn/downloads?models=wenet&version=wenetspeech_u2pp_conformer_exp.tar.gz
  wget -O 0.wav https://huggingface.co/openspeech/wenet-models/resolve/main/zh.wav
  soxi 0.wav

  echo "Test streaming"
  ./export-onnx-streaming.py
  ls -lh
  ./test-onnx-streaming.py

  echo "Test non-streaming"
  ./export-onnx.py
  ls -lh
  ./test-onnx.py

  cat > README.md <<EOF
# Introduction
This model is converted from https://wenet.org.cn/downloads?models=wenet&version=wenetspeech_u2pp_conformer_exp.tar.gz
EOF

  popd
}

function librispeech() {
  echo "librispeech"
  wget -q https://huggingface.co/openspeech/wenet-models/resolve/main/librispeech_u2pp_conformer_exp.tar.gz
  tar xvf librispeech_u2pp_conformer_exp.tar.gz
  rm -v librispeech_u2pp_conformer_exp.tar.gz

  pushd librispeech_u2pp_conformer_exp
  mkdir -p data/train_960
  cp global_cmvn ./data/train_960
  cp ../*.py .

  export WENET_URL=https://wenet.org.cn/downloads?models=wenet&version=librispeech_u2pp_conformer_exp.tar.gz
  wget -O 0.wav https://huggingface.co/openspeech/wenet-models/resolve/main/en.wav
  soxi 0.wav

  echo "Test streaming"
  ./export-onnx-streaming.py
  ls -lh
  ./test-onnx-streaming.py

  echo "Test non-streaming"
  ./export-onnx.py
  ls -lh
  ./test-onnx.py

  cat > README.md <<EOF
# Introduction
This model is converted from https://wenet.org.cn/downloads?models=wenet&version=librispeech_u2pp_conformer_exp.tar.gz
EOF

  popd
}

function gigaspeech() {
  echo "gigaspeech"
  wget -q https://huggingface.co/openspeech/wenet-models/resolve/main/gigaspeech_u2pp_conformer_exp.tar.gz
  tar xvf gigaspeech_u2pp_conformer_exp.tar.gz
  rm -v gigaspeech_u2pp_conformer_exp.tar.gz

  pushd 20210728_u2pp_conformer_exp
  mkdir -p data/gigaspeech_train_xl
  cp global_cmvn ./data/gigaspeech_train_xl
  cp ../*.py .

  export WENET_URL=https://wenet.org.cn/downloads?models=wenet&version=gigaspeech_u2pp_conformer_exp.tar.gz
  wget -O 0.wav https://huggingface.co/openspeech/wenet-models/resolve/main/en.wav
  soxi 0.wav

  echo "Test streaming"
  ./export-onnx-streaming.py
  ls -lh
  ./test-onnx-streaming.py

  echo "Test non-streaming"
  ./export-onnx.py
  ls -lh
  ./test-onnx.py

  cat > README.md <<EOF
# Introduction
This model is converted from https://wenet.org.cn/downloads?models=wenet&version=gigaspeech_u2pp_conformer_exp.tar.gz
EOF

  popd
}

install_dependencies

aishell

aishell2

multi_cn

wenetspeech

librispeech

gigaspeech

tree .
