# Introduction

Please refer to
https://www.modelscope.cn/models/pkufool/sherpa-onnx-kws-zipformer-wenetspeech-3.3M-2024-01-01/summary
to download a model.

# Kws

The following is an example:
```bash
cd sherpa-onnx/wasm/kws/assets
wget https://github.com/k2-fsa/sherpa-onnx/releases/download/kws-models/sherpa-onnx-kws-zipformer-wenetspeech-3.3M-2024-01-01.tar.bz2
tar xvf sherpa-onnx-kws-zipformer-wenetspeech-3.3M-2024-01-01.tar.bz2
rm sherpa-onnx-kws-zipformer-wenetspeech-3.3M-2024-01-01.tar.bz2

mv sherpa-onnx-kws-zipformer-wenetspeech-3.3M-2024-01-01/encoder-epoch-12-avg-2-chunk-16-left-64.onnx ./
mv sherpa-onnx-kws-zipformer-wenetspeech-3.3M-2024-01-01/decoder-epoch-12-avg-2-chunk-16-left-64.onnx ./
mv sherpa-onnx-kws-zipformer-wenetspeech-3.3M-2024-01-01/joiner-epoch-12-avg-2-chunk-16-left-64.onnx ./
mv sherpa-onnx-kws-zipformer-wenetspeech-3.3M-2024-01-01/tokens.txt ./
rm -rf sherpa-onnx-kws-zipformer-wenetspeech-3.3M-2024-01-01
```

You should have the following files in `assets` before you can run
`build-wasm-simd-kws.sh`

```bash
fangjuns-MacBook-Pro:assets fangjun$ pwd
/Users/fangjun/open-source/sherpa-onnx/wasm/kws/assets

fangjuns-MacBook-Pro:assets fangjun$ ls -lh
total 25616
-rw-r--r--  1 fangjun  staff   692B Oct 29 16:53 README.md
-rw-r--r--  1 fangjun  staff   660K Aug 14 15:21 decoder-epoch-12-avg-2-chunk-16-left-64.onnx
-rw-r--r--  1 fangjun  staff    12M Aug 14 15:21 encoder-epoch-12-avg-2-chunk-16-left-64.onnx
-rw-r--r--  1 fangjun  staff   247K Aug 14 15:21 joiner-epoch-12-avg-2-chunk-16-left-64.onnx
-rw-r--r--  1 fangjun  staff   1.6K Aug 14 15:08 tokens.txt
```

**Hint**: Remember to remove extra files from ``assets``. For instance, please remember to remove
the file `sherpa-onnx-kws-zipformer-wenetspeech-3.3M-2024-01-01.tar.bz2`.
