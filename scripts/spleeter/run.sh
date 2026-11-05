#!/usr/bin/env bash
#
# Usage:  ./run.sh [2stems|4stems]      (default: 2stems)
#
# 4stems is not simply "the same thing with more stems". Spleeter's
# configs/4stems/base_config.json asks for ELU activations where 2stems takes
# the unet.unet defaults (LeakyReLU/ReLU). The architecture and every weight
# shape are identical, so the wrong activation loads without complaint and
# quietly returns garbage. convert_to_torch.py's ACTIVATIONS table keeps the
# two apart.

set -e

model=${1:-2stems}

case "$model" in
  2stems) stems="vocals accompaniment" ;;
  4stems) stems="vocals drums bass other" ;;
  *) echo "usage: $0 [2stems|4stems]" >&2; exit 1 ;;
esac

if [ ! -f $model.tar.gz ]; then
  curl -SL -O https://github.com/deezer/spleeter/releases/download/v1.4.0/$model.tar.gz
fi

if [ ! -d ./$model ]; then
  mkdir -p $model
  cd $model
  tar xvf ../$model.tar.gz
  cd ..
fi

ls -lh

ls -lh $model

for s in $stems; do
  if [ ! -f $model/frozen_${s}_model.pb ]; then
    python3 ./convert_to_pb.py \
      --model-dir ./$model \
      --output-node-names ${s}_spectrogram/mul \
      --output-filename ./$model/frozen_${s}_model.pb
  fi

  ls -lh $model

  python3 ./convert_to_torch.py --model $model --name $s
done

python3 ./export_onnx.py --model $model

ls -lh $model
