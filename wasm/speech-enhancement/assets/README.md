# Introduction

## Huggingface space

You can visit https://huggingface.co/spaces/k2-fsa/wasm-speech-enhancement-gtcrn
to try it in your browser without building or installing anything.

You can also visit
https://modelscope.cn/studios/csukuangfj/wasm-speech-enhancement-gtcrn

## Usage

Please refer to
https://github.com/k2-fsa/sherpa-onnx/releases/tag/speech-enhancement-models
to download a model.

The following is an example:

```bash
cd sherpa-onnx/wasm/speech-enhancement/assets
wget https://github.com/k2-fsa/sherpa-onnx/releases/download/speech-enhancement-models/gtcrn_simple.onnx

mv gtcrn_simple.onnx gtcrn.onnx
```

You should have the following files in `assets` before you can run
`build-wasm-simd-speech-enhancement.sh`

```
(py38) fangjuns-MacBook-Pro:assets fangjun$ tree .
.
├── README.md
└── gtcrn.onnx

0 directories, 2 files
(py38) fangjuns-MacBook-Pro:assets fangjun$ ls -lh
total 1056
-rw-r--r--  1 fangjun  staff   466B Mar 12 16:13 README.md
-rw-r--r--  1 fangjun  staff   523K Mar 12 16:14 gtcrn.onnx
```
