#!/usr/bin/env bash

if [ -z $t ]; then
  echo "Please run export t=num_input_seconds"
  exit -1
fi

if [ -z $soc ]; then
  echo "Please run export soc=SM8850, etc."
  exit -1
fi

if [ -z $QNN_SDK_ROOT ]; then
  echo "Please run setup QNN first"
  exit -1
fi

echo "Export to onnx with num_seconds $t"

python3 ./export_encoder_onnx.py --input-len-in-seconds $t --opset-version 17

ls -lh encoder-*.onnx

python3 ../../pyannote/segmentation/show-onnx.py --filename ./encoder-$t-seconds.onnx

echo "Generate test data"

python3 ./generate_encoder_data.py --input-len-in-seconds $t

ls -lh encoder-*

echo "---"
cat ./encoder-input-list.txt
echo "---"

echo "Convert onnx to qnn"


qnn-onnx-converter \
  --input_network ./encoder-$t-seconds.onnx \
  --output_path ./encoder-$t-seconds-quantized \
  --out_node encoder_out \
  --input_list ./encoder-input-list.txt \
  --use_native_input_files  \
  --input_dtype x float32 \
  --act_bitwidth 16 \
  --bias_bitwidth 32 \
  --input_layout x NTF

ls -lh

mv -v encoder-$t-seconds-quantized encoder-$t-seconds-quantized.cpp

python3 ../../qnn/generate_config.py \
    --soc $soc \
    --graph-name "encoder_${t}_seconds_quantized" \
    --output-dir ./my-config \
    --qnn-sdk-root $QNN_SDK_ROOT

ls -lh my-config

head -n100 ./my-config/*.json

python3 "${QNN_SDK_ROOT}/bin/x86_64-linux-clang/qnn-model-lib-generator" \
    -c "encoder-$t-seconds-quantized.cpp" \
    -b "encoder-$t-seconds-quantized.bin" \
    -o model_libs
    # -t x86_64-linux-clang \

ls -lh model_libs/x86_64-linux-clang/

$QNN_SDK_ROOT/bin/x86_64-linux-clang/qnn-context-binary-generator \
  --backend $QNN_SDK_ROOT/lib/x86_64-linux-clang/libQnnHtp.so \
  --model ./model_libs/x86_64-linux-clang/libencoder-${t}-seconds-quantized.so \
  --output_dir ./binary \
  --binary_file encoder \
  --config_file ./my-config/htp_backend_extensions.json

ls -lh binary

echo "Finish exporting encoder"
