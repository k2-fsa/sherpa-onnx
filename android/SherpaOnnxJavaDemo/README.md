# Introduction

Please run the following commands to download model files before you run this Android demo:

```bash
# Assume we are inside
# /Users/fangjun/open-source/sherpa-onnx/android/SherpaOnnxJavaDemo

cd app/src/main/assets/
wget https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20.tar.bz2

tar xvf sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20.tar.bz2
rm sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20.tar.bz2

mv sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20/encoder-epoch-99-avg-1.int8.onnx ./
mv sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20/decoder-epoch-99-avg-1.onnx ./
mv sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20/joiner-epoch-99-avg-1.int8.onnx ./
mv sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20/tokens.txt ./

rm -rf sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20/*

mv encoder-epoch-99-avg-1.int8.onnx sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20/
mv decoder-epoch-99-avg-1.onnx sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20/
mv joiner-epoch-99-avg-1.int8.onnx sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20/
mv tokens.txt sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20/
```

You should have the following directory structure:
```
(py38) fangjuns-MacBook-Pro:assets fangjun$ pwd
/Users/fangjun/open-source/sherpa-onnx/android/SherpaOnnxJavaDemo/app/src/main/assets

(py38) fangjuns-MacBook-Pro:assets fangjun$ tree .
.
└── sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20
    ├── decoder-epoch-99-avg-1.onnx
    ├── encoder-epoch-99-avg-1.int8.onnx
    ├── joiner-epoch-99-avg-1.int8.onnx
    └── tokens.txt

1 directory, 4 files
```

Remember to remove unused files to reduce the file size of the final APK.
