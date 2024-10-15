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

**WARNING**: Tested with `onnxruntime==1.16.3 onnx==1.15.0`.

```bash
pip install onnxruntime==1.16.3 onnx==1.15.0
```

## More examples

### [sherpa-onnx-streaming-zipformer-korean-2024-06-16](https://k2-fsa.github.io/sherpa/onnx/pretrained_models/online-transducer/zipformer-transducer-models.html#sherpa-onnx-streaming-zipformer-korean-2024-06-16-korean)


| | encoder-epoch-99-avg-1.onnx | encoder-epoch-99-avg-1.int8.onnx|
|---|---|---|
|Dynamic batch size| 279 MB| 122 MB|
|Batch size fixed to 1| 264 MB | 107 MB |

### [sherpa-onnx-streaming-zipformer-en-20M-2023-02-17](https://k2-fsa.github.io/sherpa/onnx/pretrained_models/online-transducer/zipformer-transducer-models.html#csukuangfj-sherpa-onnx-streaming-zipformer-en-20m-2023-02-17-english)

| | encoder-epoch-99-avg-1.onnx | encoder-epoch-99-avg-1.int8.onnx|
|---|---|---|
|Dynamic batch size| 85 MB| 41 MB|
|Batch size fixed to 1| 75 MB | 32 MB |

### [sherpa-onnx-streaming-zipformer-multi-zh-hans-2023-12-12](https://k2-fsa.github.io/sherpa/onnx/pretrained_models/online-transducer/zipformer-transducer-models.html#sherpa-onnx-streaming-zipformer-multi-zh-hans-2023-12-12-chinese)

| | encoder-epoch-20-avg-1-chunk-16-left-128.onnx | encoder-epoch-20-avg-1-chunk-16-left-128.int8.onnx|
|---|---|---|
|Dynamic batch size| 249 MB| 67 MB|
|Batch size fixed to 1| 247 MB | 65 MB |

### [icefall-asr-zipformer-streaming-wenetspeech-20230615](https://k2-fsa.github.io/sherpa/onnx/pretrained_models/online-transducer/zipformer-transducer-models.html#pkufool-icefall-asr-zipformer-streaming-wenetspeech-20230615-chinese)

| | encoder-epoch-12-avg-4-chunk-16-left-128.onnx | encoder-epoch-12-avg-4-chunk-16-left-128.int8.onnx|
|---|---|---|
|Dynamic batch size| 250 MB| 68 MB|
|Batch size fixed to 1| 247 MB | 65 MB |

### [sherpa-onnx-streaming-zipformer-en-2023-06-26](https://k2-fsa.github.io/sherpa/onnx/pretrained_models/online-transducer/zipformer-transducer-models.html#csukuangfj-sherpa-onnx-streaming-zipformer-en-2023-06-26-english)


| | encoder-epoch-99-avg-1-chunk-16-left-128.onnx | encoder-epoch-99-avg-1-chunk-16-left-128.int8.onnx|
|---|---|---|
|Dynamic batch size| 250 MB| 68 MB|
|Batch size fixed to 1| 247 MB | 65 MB |

### [sherpa-onnx-streaming-zipformer-en-2023-06-21](https://k2-fsa.github.io/sherpa/onnx/pretrained_models/online-transducer/zipformer-transducer-models.html#csukuangfj-sherpa-onnx-streaming-zipformer-en-2023-06-21-english)

| | encoder-epoch-99-avg-1.onnx | encoder-epoch-99-avg-1.int8.onnx|
|---|---|---|
|Dynamic batch size| 338 MB| 180 MB|
|Batch size fixed to 1| 264 MB | 107 MB |

### [sherpa-onnx-streaming-zipformer-en-2023-02-21](https://k2-fsa.github.io/sherpa/onnx/pretrained_models/online-transducer/zipformer-transducer-models.html#csukuangfj-sherpa-onnx-streaming-zipformer-en-2023-02-21-english)

| | encoder-epoch-99-avg-1.onnx | encoder-epoch-99-avg-1.int8.onnx|
|---|---|---|
|Dynamic batch size| 279 MB| 122 MB|
|Batch size fixed to 1| 264 MB | 107 MB |

### [sherpa-onnx-streaming-zipformer-fr-2023-04-14](https://k2-fsa.github.io/sherpa/onnx/pretrained_models/online-transducer/zipformer-transducer-models.html#shaojieli-sherpa-onnx-streaming-zipformer-fr-2023-04-14-french)

| | encoder-epoch-29-avg-9-with-averaged-model.onnx | encoder-epoch-29-avg-9-with-averaged-model.int8.onnx|
|---|---|---|
|Dynamic batch size| 279 MB| 121 MB|
|Batch size fixed to 1| 264 MB | 107 MB |

### [sherpa-onnx-streaming-zipformer-small-bilingual-zh-en-2023-02-16](https://k2-fsa.github.io/sherpa/onnx/pretrained_models/online-transducer/zipformer-transducer-models.html#sherpa-onnx-streaming-zipformer-small-bilingual-zh-en-2023-02-16-bilingual-chinese-english)

| | encoder-epoch-99-avg-1.onnx | encoder-epoch-99-avg-1.int8.onnx|
|---|---|---|
|Dynamic batch size| 85 MB| 41 MB|
|Batch size fixed to 1| 75 MB | 32 MB |

### [sherpa-onnx-streaming-zipformer-zh-14M-2023-02-23](https://k2-fsa.github.io/sherpa/onnx/pretrained_models/online-transducer/zipformer-transducer-models.html#csukuangfj-sherpa-onnx-streaming-zipformer-zh-14m-2023-02-23-chinese)

| | encoder-epoch-99-avg-1.onnx | encoder-epoch-99-avg-1.int8.onnx|
|---|---|---|
|Dynamic batch size| 40 MB| 21 MB|
|Batch size fixed to 1| 33 MB | 15 MB |

### [sherpa-onnx-kws-zipformer-wenetspeech-3.3M-2024-01-01](https://k2-fsa.github.io/sherpa/onnx/kws/pretrained_models/index.html#sherpa-onnx-kws-zipformer-wenetspeech-3-3m-2024-01-01-chinese)

| | encoder-epoch-12-avg-2-chunk-16-left-64.onnx | encoder-epoch-12-avg-2-chunk-16-left-64.int8.onnx|
|---|---|---|
|Dynamic batch size| 12 MB| 4.6 MB|
|Batch size fixed to 1| 11 MB | 3.9 MB |

### [sherpa-onnx-kws-zipformer-gigaspeech-3.3M-2024-01-01](https://k2-fsa.github.io/sherpa/onnx/kws/pretrained_models/index.html#sherpa-onnx-kws-zipformer-gigaspeech-3-3m-2024-01-01-english)

| | encoder-epoch-12-avg-2-chunk-16-left-64.onnx | encoder-epoch-12-avg-2-chunk-16-left-64.int8.onnx|
|---|---|---|
|Dynamic batch size| 12 MB| 4.6 MB|
|Batch size fixed to 1| 11 MB | 3.9 MB |
