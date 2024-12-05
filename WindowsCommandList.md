# Windows Install :[93 新一代 Kaldi 部署框架 sherpa-onnx 之 Windows 安装 (适合0基础，手把手教)_哔哩哔哩_bilibili](https://www.bilibili.com/video/BV1Um421V75A?spm_id_from=333.788.videopod.sections&vd_source=964bbd88f350a12d2453698dd08ec8ca)

# Below The Model used is : . \sherpa-onnx\sherpa-onnx-paraformer-zh-small-2024-03-09

### use the test wav files to test the model:

```
./build/bin/Release/sherpa-onnx-offline `
--tokens=./sherpa-onnx-paraformer-zh-small-2024-03-09/tokens.txt `
--paraformer=./sherpa-onnx-paraformer-zh-small-2024-03-09/model.int8.onnx `
./sherpa-onnx-paraformer-zh-small-2024-03-09/test_wavs/0.wav `
./sherpa-onnx-paraformer-zh-small-2024-03-09/test_wavs/1.wav `
./sherpa-onnx-paraformer-zh-small-2024-03-09/test_wavs/8k.wav `
./sherpa-onnx-paraformer-zh-small-2024-03-09/test_wavs/2-zh-en.wav `
./sherpa-onnx-paraformer-zh-small-2024-03-09/test_wavs/3-sichuan.wav `
./sherpa-onnx-paraformer-zh-small-2024-03-09/test_wavs/4-tianjin.wav `
./sherpa-onnx-paraformer-zh-small-2024-03-09/test_wavs/5-henan.wav
```

### use the microphone to test the on time offline model:

```
./build/bin/Release/sherpa-onnx-microphone-offline `
--tokens=./sherpa-onnx-paraformer-zh-small-2024-03-09/tokens.txt `
--paraformer=./sherpa-onnx-paraformer-zh-small-2024-03-09/model.int8.onnx
```

# Below The Model used is : ./sherpa-onnx-streaming-paraformer-bilingual-zh-en

### Use the Micriphone to test the Realtime stream:

```
./build/bin/Release/sherpa-onnx-microphone `
--tokens=./sherpa-onnx-streaming-paraformer-bilingual-zh-en/tokens.txt `
--paraformer-encoder=./sherpa-onnx-streaming-paraformer-bilingual-zh-en/encoder.int8.onnx `
--paraformer-decoder=./sherpa-onnx-streaming-paraformer-bilingual-zh-en/decoder.int8.onnx
```

# Below The Model used is : .\sherpa-onnx\sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20

### Start the Server:

```
./build/bin/Release/sherpa-onnx-online-websocket-server `
  --port=6006 `
  --num-work-threads=3 `
  --num-io-threads=2 `
  --tokens=./sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20/tokens.txt `
  --encoder=./sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20/encoder-epoch-99-avg-1.onnx `
  --decoder=./sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20/decoder-epoch-99-avg-1.onnx `
  --joiner=./sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20/joiner-epoch-99-avg-1.onnx `
  --log-file=./log.txt `
  --max-batch-size=5 `
  --loop-interval-ms=20

```

### Start the Python Microphone Client: python-api-examples\online-websocket-client-microphone.py

```
python3 ./python-api-examples/online-websocket-client-decode-file.py `
  --server-addr localhost `
  --server-port 6006 `
  --seconds-per-message 0.1 `
  ./sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20/test_wavs/4.wav

```

```
# Q&A

### In Windows, if the output is wired, you can try to use the following command to test the model:

[Frequently Asked Question (FAQs) — sherpa 1.3 documentation](https://k2-fsa.github.io/sherpa/onnx/tts/faq.html) & command: CHCP65001
```
