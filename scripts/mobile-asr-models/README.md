# Introduction

This folder contains scripts to convert ASR models for mobile platforms
supporting only batch size equal to 1.

The advantage of fixing the batch size to 1 is that it provides more
opportunities for model optimization and quantization.

To give you a concrete example, for the following model
https://k2-fsa.github.io/sherpa/onnx/pretrained_models/online-transducer/zipformer-transducer-models.html#csukuangfj-sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20-bilingual-chinese-english

| | encoder-epoch-99-avg-1.onnx | encoder-epoch-99-avg-1.int8.onnx|
|---|---|---|
|Dynamic batch size| 315 MB| 174 MB|
|Batch size fixed to 1| 242 MB | 100 MB |

The following [colab notebook](https://colab.research.google.com/drive/1RsVZbsxbPjazeGrNNbZNjXCYbEG2F2DU?usp=sharing)
provides examples to use the above two models.
