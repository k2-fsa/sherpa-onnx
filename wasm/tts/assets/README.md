# Introduction

Please refer to
https://github.com/k2-fsa/sherpa-onnx/releases/tag/tts-models
to download a model.

The following is an example:
```
cd sherpa-onnx/wasm/tts/assets

wget -q https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/vits-piper-en_US-libritts_r-medium.tar.bz2
tar xf vits-piper-en_US-libritts_r-medium.tar.bz2
rm vits-piper-en_US-libritts_r-medium.tar.bz2
mv vits-piper-en_US-libritts_r-medium/en_US-libritts_r-medium.onnx ./model.onnx
mv vits-piper-en_US-libritts_r-medium/tokens.txt ./
mv vits-piper-en_US-libritts_r-medium/espeak-ng-data ./
rm -rf vits-piper-en_US-libritts_r-medium
```

You should have the following files in `assets` before you can run
`build-wasm-simd-tts.sh`

```
assets fangjun$ tree -L 1
.
├── README.md
├── espeak-ng-data
├── mode.onnx
└── tokens.txt

1 directory, 3 files
```
