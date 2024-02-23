# Introduction

Please refer to
https://github.com/k2-fsa/sherpa-onnx/releases/tag/asr-models
or
https://k2-fsa.github.io/sherpa/onnx/pretrained_models/index.html
to download a model.

# Streaming ASR

## Transducer
```bash
cd sherpa-onnx/wasm/asr/assets

wget -q https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20.tar.bz2
tar xvf sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20.tar.bz2
rm sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20.tar.bz2

# Note it is not an error that we rename encoder.int8.onnx to encoder.onnx

mv sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20/encoder-epoch-99-avg-1.int8.onnx encoder.onnx
mv sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20/decoder-epoch-99-avg-1.onnx decoder.onnx
mv sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20/joiner-epoch-99-avg-1.int8.onnx joiner.onnx
mv sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20/tokens.txt ./
rm -rf sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20/

cd ../../..

./build-wasm-simd-asr.sh
```

You should have the following files in `assets` before you can run
`build-wasm-simd-asr.sh`

```
assets fangjun$ tree -L 1
.
├── README.md
├── decoder.onnx
├── encoder.onnx
├── joiner.onnx
└── tokens.txt

0 directories, 5 files
```

## Paraformer

```
wget https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-streaming-paraformer-bilingual-zh-en.tar.bz2
tar xvf sherpa-onnx-streaming-paraformer-bilingual-zh-en.tar.bz2
rm sherpa-onnx-streaming-paraformer-bilingual-zh-en.tar.bz2

mv sherpa-onnx-streaming-paraformer-bilingual-zh-en/encoder.int8.onnx encoder.onnx
mv sherpa-onnx-streaming-paraformer-bilingual-zh-en/decoder.int8.onnx decoder.onnx
mv sherpa-onnx-streaming-paraformer-bilingual-zh-en/tokens.txt ./

rm -rf sherpa-onnx-streaming-paraformer-bilingual-zh-en

cd ../

sed -i.bak s/"type = 0"/"type = 1"/g ./sherpa-onnx.js
sed -i.bak s/Zipformer/Paraformer/g ./index.html

cd ../..

./build-wasm-simd-asr.sh
```

You should have the following files in `assets` before you can run
`build-wasm-simd-asr.sh`

```
assets fangjun$ tree -L 1
.
├── README.md
├── decoder.onnx
├── encoder.onnx
└── tokens.txt

0 directories, 4 files
```
