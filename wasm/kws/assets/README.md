# Introduction

Please refer to
https://www.modelscope.cn/models/pkufool/sherpa-onnx-kws-zipformer-wenetspeech-3.3M-2024-01-01/summary
to download a model.

# Kws

The following is an example:
```
cd sherpa-onnx/wasm/kws
git clone https://www.modelscope.cn/pkufool/sherpa-onnx-kws-zipformer-wenetspeech-3.3M-2024-01-01.git assets
```

You should have the following files in `assets` before you can run
`build-wasm-simd-kws.sh`

```
├── decoder-epoch-12-avg-2-chunk-16-left-64.onnx
├── encoder-epoch-12-avg-2-chunk-16-left-64.onnx
├── joiner-epoch-12-avg-2-chunk-16-left-64.onnx
├── keywords_raw.txt
├── keywords.txt
├── README.md
└── tokens.txt

```
