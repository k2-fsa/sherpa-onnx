#!/usr/bin/env bash


if [ ! -f 2stems.tar.gz ]; then
  wget https://github.com/deezer/spleeter/releases/download/v1.4.0/2stems.tar.gz
fi

if [ ! -d ./2stems ]; then
  mkdir -p 2stems
  cd 2stems
  tar xvf ../2stems.tar.gz
  cd ..
fi

if [ ! -f 2stems/frozen_vocals_model.pb ]; then
  python3 ./convert_to_pb.py \
    --model-dir ./2stems \
    --output-node-names vocals_spectrogram/mul \
    --output-filename ./2stems/frozen_vocals_model.pb
fi

if [ ! -f 2stems/frozen_accompaniment_model.pb ]; then
  python3 ./convert_to_pb.py \
    --model-dir ./2stems \
    --output-node-names accompaniment_spectrogram/mul \
    --output-filename ./2stems/frozen_accompaniment_model.pb
fi

python3 ./convert_to_torch.py --name vocals
python3 ./convert_to_torch.py --name accompaniment
