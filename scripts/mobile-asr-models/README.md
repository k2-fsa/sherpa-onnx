# Introduction

This folder contains scripts to convert ASR models for mobile platforms
where it supports only batch size equal to 1.

The advantage of fixing the batch size to 1 is that it provides more
opportunities for model optimization and quantization.

To give you a concrete example, for the following model
https://k2-fsa.github.io/sherpa/onnx/pretrained_models/online-transducer/zipformer-transducer-models.html#csukuangfj-sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20-bilingual-chinese-english

| | encoder-epoch-99-avg-1.onnx | encoder-epoch-99-avg-1.int8.onnx| Ratio|
|Dynamic batch size| 315 MB| 174 MB| 315/174 = 1.81|
|Fix batch size to 1| 315 MB | 100 MB | 315/100 = 3.15|
