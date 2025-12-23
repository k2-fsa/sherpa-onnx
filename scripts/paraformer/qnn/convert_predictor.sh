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

python3 ./export_predictor_onnx.py --input-len-in-seconds $t --opset-version 17

ls -lh predictor-*.onnx

python3 ../../pyannote/segmentation/show-onnx.py --filename ./predictor-$t-seconds.onnx

echo "Generate test data"

python3 ./generate_predictor_data.py --input-len-in-seconds $t

ls -lh predictor-*

echo "---"
cat ./predictor-input-list.txt
echo "---"

echo "Convert onnx to qnn"


qnn-onnx-converter \
  --input_network ./predictor-$t-seconds.onnx \
  --output_path ./predictor-$t-seconds-quantized \
  --input_list ./predictor-input-list.txt \
  --use_native_input_files  \
  --input_dtype encoder_out float32 \
  --act_bitwidth 16 \
  --bias_bitwidth 32

  # Note(fangjun): It throws an error if we specify the layout for predictor input.
  # --input_layout encoder_out NTF

ls -lh

mv -v predictor-$t-seconds-quantized predictor-$t-seconds-quantized.cpp

python3 ../../qnn/generate_config.py \
    --soc $soc \
    --graph-name "predictor_${t}_seconds_quantized" \
    --output-dir ./my-config-2 \
    --qnn-sdk-root $QNN_SDK_ROOT

ls -lh my-config-2

head -n100 ./my-config-2/*.json

python3 "${QNN_SDK_ROOT}/bin/x86_64-linux-clang/qnn-model-lib-generator" \
    -c "predictor-$t-seconds-quantized.cpp" \
    -b "predictor-$t-seconds-quantized.bin" \
    -o model_libs
    # -t x86_64-linux-clang \

ls -lh model_libs/x86_64-linux-clang/

$QNN_SDK_ROOT/bin/x86_64-linux-clang/qnn-context-binary-generator \
  --backend $QNN_SDK_ROOT/lib/x86_64-linux-clang/libQnnHtp.so \
  --model ./model_libs/x86_64-linux-clang/libpredictor-${t}-seconds-quantized.so \
  --output_dir ./binary \
  --binary_file predictor \
  --config_file ./my-config-2/htp_backend_extensions.json

ls -lh binary

echo "Finish exporting predictor"
