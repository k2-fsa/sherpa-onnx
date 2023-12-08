#!/usr/bin/env bash

set -ex

echo "Downloading models"
export GIT_LFS_SKIP_SMUDGE=1
git clone https://huggingface.co/openspeech/wespeaker-models
cd wespeaker-models
git lfs pull --include "*.onnx"
ls -lh
cd ..
mv wespeaker-models/*.onnx .
ls -lh

./add_meta_data.py \
  --model ./voxceleb_resnet34.onnx \
  --language English \
  --url https://wespeaker-1256283475.cos.ap-shanghai.myqcloud.com/models/voxceleb/voxceleb_resnet34.onnx
./test.py  --model ./voxceleb_resnet34.onnx \
  --file1 ./wespeaker-models/test_wavs/00001_spk1.wav \
  --file2 ./wespeaker-models/test_wavs/00024_spk1.wav \

./test.py  --model ./voxceleb_resnet34.onnx \
  --file1 ./wespeaker-models/test_wavs/00001_spk1.wav \
  --file2 ./wespeaker-models/test_wavs/00010_spk2.wav

mv voxceleb_resnet34.onnx en_voxceleb_resnet34.onnx

./add_meta_data.py \
  --model ./voxceleb_resnet34_LM.onnx \
  --language English \
  --url https://wespeaker-1256283475.cos.ap-shanghai.myqcloud.com/models/voxceleb/voxceleb_resnet34_LM.onnx
./test.py  --model ./voxceleb_resnet34_LM.onnx \
  --file1 ./wespeaker-models/test_wavs/00001_spk1.wav \
  --file2 ./wespeaker-models/test_wavs/00024_spk1.wav \

./test.py  --model ./voxceleb_resnet34_LM.onnx \
  --file1 ./wespeaker-models/test_wavs/00001_spk1.wav \
  --file2 ./wespeaker-models/test_wavs/00010_spk2.wav

mv voxceleb_resnet34_LM.onnx en_voxceleb_resnet34_LM.onnx

./add_meta_data.py \
  --model ./voxceleb_resnet152_LM.onnx \
  --language English \
  --url https://wespeaker-1256283475.cos.ap-shanghai.myqcloud.com/models/voxceleb/voxceleb_resnet152_LM.onnx

./test.py  --model ./voxceleb_resnet152_LM.onnx \
  --file1 ./wespeaker-models/test_wavs/00001_spk1.wav \
  --file2 ./wespeaker-models/test_wavs/00024_spk1.wav \

./test.py  --model ./voxceleb_resnet152_LM.onnx \
  --file1 ./wespeaker-models/test_wavs/00001_spk1.wav \
  --file2 ./wespeaker-models/test_wavs/00010_spk2.wav

mv voxceleb_resnet152_LM.onnx en_voxceleb_resnet152_LM.onnx

./add_meta_data.py \
  --model ./voxceleb_resnet221_LM.onnx \
  --language English \
  --url https://wespeaker-1256283475.cos.ap-shanghai.myqcloud.com/models/voxceleb/voxceleb_resnet221_LM.onnx

./test.py  --model ./voxceleb_resnet221_LM.onnx \
  --file1 ./wespeaker-models/test_wavs/00001_spk1.wav \
  --file2 ./wespeaker-models/test_wavs/00024_spk1.wav \

./test.py  --model ./voxceleb_resnet221_LM.onnx \
  --file1 ./wespeaker-models/test_wavs/00001_spk1.wav \
  --file2 ./wespeaker-models/test_wavs/00010_spk2.wav

mv voxceleb_resnet221_LM.onnx en_voxceleb_resnet221_LM.onnx

./add_meta_data.py \
  --model ./voxceleb_resnet293_LM.onnx \
  --language English \
  --url https://wespeaker-1256283475.cos.ap-shanghai.myqcloud.com/models/voxceleb/voxceleb_resnet293_LM.onnx

./test.py  --model ./voxceleb_resnet293_LM.onnx \
  --file1 ./wespeaker-models/test_wavs/00001_spk1.wav \
  --file2 ./wespeaker-models/test_wavs/00024_spk1.wav \

./test.py  --model ./voxceleb_resnet293_LM.onnx \
  --file1 ./wespeaker-models/test_wavs/00001_spk1.wav \
  --file2 ./wespeaker-models/test_wavs/00010_spk2.wav

mv voxceleb_resnet293_LM.onnx en_voxceleb_resnet293_LM.onnx

./add_meta_data.py \
  --model ./voxceleb_CAM++.onnx \
  --language English \
  --url https://wespeaker-1256283475.cos.ap-shanghai.myqcloud.com/models/voxceleb/voxceleb_CAM++.onnx

./test.py  --model ./voxceleb_CAM++.onnx \
  --file1 ./wespeaker-models/test_wavs/00001_spk1.wav \
  --file2 ./wespeaker-models/test_wavs/00024_spk1.wav \

./test.py  --model ./voxceleb_CAM++.onnx \
  --file1 ./wespeaker-models/test_wavs/00001_spk1.wav \
  --file2 ./wespeaker-models/test_wavs/00010_spk2.wav

mv voxceleb_CAM++.onnx en_voxceleb_CAM++.onnx

./add_meta_data.py \
  --model ./voxceleb_CAM++_LM.onnx \
  --language English \
  --url https://wespeaker-1256283475.cos.ap-shanghai.myqcloud.com/models/voxceleb/voxceleb_CAM++_LM.onnx

./test.py  --model ./voxceleb_CAM++_LM.onnx \
  --file1 ./wespeaker-models/test_wavs/00001_spk1.wav \
  --file2 ./wespeaker-models/test_wavs/00024_spk1.wav \

./test.py  --model ./voxceleb_CAM++_LM.onnx \
  --file1 ./wespeaker-models/test_wavs/00001_spk1.wav \
  --file2 ./wespeaker-models/test_wavs/00010_spk2.wav

mv voxceleb_CAM++_LM.onnx en_voxceleb_CAM++_LM.onnx

./add_meta_data.py \
  --model ./cnceleb_resnet34.onnx \
  --language Chinese \
  --url https://wespeaker-1256283475.cos.ap-shanghai.myqcloud.com/models/cnceleb/cnceleb_resnet34.onnx

mv cnceleb_resnet34.onnx zh_cnceleb_resnet34.onnx

./add_meta_data.py \
  --model ./cnceleb_resnet34_LM.onnx \
  --language Chinese \
  --url https://wespeaker-1256283475.cos.ap-shanghai.myqcloud.com/models/cnceleb/cnceleb_resnet34_LM.onnx

mv cnceleb_resnet34_LM.onnx zh_cnceleb_resnet34_LM.onnx

ls -lh
