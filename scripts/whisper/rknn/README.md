# Usage

You can find pre-exported rknn models for rk3588 at

https://modelscope.cn/models/csukuangfj/2026-01-05-rknn/files


# Download test wave

```
wget https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/en.wav
wget https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/en-16k.wav
```

## Export to onnx

```
./export_onnx.py --model tiny.en
```

## Test onnx

```
./test_onnx.py --model tiny.en
```

## Export to rknn

```
python3 ./export_rknn.py --target-platform rk3588  --in-model ./tiny.en-encoder.onnx --out-model ./tiny.en-encoder.rknn

python3 ./export_rknn.py --target-platform rk3588  --in-model ./tiny.en-decoder.onnx --out-model ./tiny.en-decoder.rknn
```

```
ls -lh tiny.en-*.rknn

-rw-r--r-- 1 kuangfangjun root 95M Jan  5 16:16 tiny.en-decoder.rknn
-rw-r--r-- 1 kuangfangjun root 22M Jan  5 16:15 tiny.en-encoder.rknn
```

## Run it on your rk3588 board

```
wget https://huggingface.co/csukuangfj/sherpa-onnx-whisper-tiny.en/resolve/main/tiny.en-tokens.txt

./test_on_rk3588_board.py  --encoder ./tiny.en-encoder.rknn --decoder ./tiny.en-decoder.rknn --tokens ./tiny.en-tokens.txt --wav ./en-16k.wav
```
