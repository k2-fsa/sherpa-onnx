#!/usr/bin/env bash

wget https://hf-mirror.com/csukuangfj/WSChuan-ASR/resolve/main/Paraformer-large-Chuan/am.mvn
wget https://hf-mirror.com/csukuangfj/WSChuan-ASR/resolve/main/Paraformer-large-Chuan/config.yaml
wget https://hf-mirror.com/csukuangfj/WSChuan-ASR/resolve/main/Paraformer-large-Chuan/tokens.json
wget https://hf-mirror.com/csukuangfj/WSChuan-ASR/resolve/main/Paraformer-large-Chuan/seg_dict
wget https://hf-mirror.com/csukuangfj/WSChuan-ASR/resolve/main/Paraformer-large-Chuan/model_state_dict.pt

python3 ./export_encoder_onnx.py  --input-len-in-seconds 5
python3 ./export_rknn.py --target-platform rk3588 --in-model ./encoder-5-seconds.onnx --out-model ./encoder-5-seconds.rknn

python3 ./export_predictor_onnx.py  --input-len-in-seconds 5
python3 ./export_rknn.py --target-platform rk3588 --in-model ./predictor-5-seconds.onnx --out-model ./predictor-5-seconds.rknn

python3 ./export_decoder_onnx.py  --input-len-in-seconds 5
python3 ./export_rknn.py --target-platform rk3588 --in-model ./decoder-5-seconds.onnx --out-model ./decoder-5-seconds.rknn
