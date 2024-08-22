#!/usr/bin/env bash
#
# usage of this file:
#  ./run.sh --input in.onnx --output1 out1.onnx --output2 out2.onnx
# where out1.onnx is a float32 model with batch size fixed to 1
# and out2.onnx is an int8 quantized version of out1.onnx

set -ex

input=
output1=
output2=
batch_dim=N
source ./parse_options.sh

if [ -z $input ]; then
  echo 'Please provide input model filename'
  exit 1
fi

if [ -z $output1 ]; then
  echo 'Please provide output1 model filename'
  exit 1
fi

if [ -z $output2 ]; then
  echo 'Please provide output2 model filename'
  exit 1
fi


echo "input: $input"
echo "output1: $output1"
echo "output2: $output2"

python3 -m onnxruntime.tools.make_dynamic_shape_fixed --dim_param $batch_dim --dim_value 1 $input tmp.fixed.onnx
python3 -m onnxruntime.quantization.preprocess --input tmp.fixed.onnx --output $output1
python3 ./dynamic_quantization.py --input $output1 --output $output2

ls -lh $input tmp.fixed.onnx $output1 $output2

rm tmp.fixed.onnx
