#!/usr/bin/env bash
# Copyright      2024  Xiaomi Corp.        (authors: Fangjun Kuang)


python3 -m onnxruntime.quantization.preprocess --input model.onnx --output tmp.preprocessed.onnx
mv ./tmp.preprocessed.onnx ./model.onnx
./show-onnx.py --filename ./model.onnx

<<EOF
=========./model.onnx==========
NodeArg(name='x', type='tensor(float)', shape=[1, 1, 'T'])
-----
NodeArg(name='y', type='tensor(float)', shape=[1, 'floor(floor(floor(floor(T/10 - 251/10)/3 - 2/3)/3)/3 - 8/3) + 1', 7])

  floor(floor(floor(floor(T/10 - 251/10)/3 - 2/3)/3)/3 - 8/3) + 1
= floor(floor(floor(floor(T - 251)/30 - 2/3)/3)/3 - 8/3) + 1
= floor(floor(floor(floor(T - 271)/30)/3)/3 - 8/3) + 1
= floor(floor(floor(floor(T - 271)/90))/3 - 8/3) + 1
= floor(floor(floor(T - 271)/90)/3 - 8/3) + 1
= floor(floor((T - 271)/90)/3 - 8/3) + 1
= floor(floor((T - 271)/90 - 8)/3) + 1
= floor(floor((T - 271 - 720)/90)/3) + 1
= floor(floor((T - 991)/90)/3) + 1
= floor(floor((T - 991)/270)) + 1
= (T - 991)/270 + 1
= (T - 991 + 270)/270
= (T - 721)/270

It means:
 - Number of input samples should be at least 721
 - One frame corresponds to 270 samples. (If we use T + 270, it outputs one more frame)
EOF
