# Introduction

This folder contains script for exporting
[silero_vad v4](https://github.com/snakers4/silero-vad/tree/v4.0)
to rknn.

# Steps to run

## 1. Download a jit model
You can download it from <https://github.com/snakers4/silero-vad/blob/v4.0/files/silero_vad.jit>

```bash
wget https://github.com/snakers4/silero-vad/raw/refs/tags/v4.0/files/silero_vad.jit
```

```bash
ls -lh silero_vad.jit
-rw-r--r-- 1 kuangfangjun root 1.4M Mar 30 11:04 silero_vad.jit
```

## 2. Export it to onnx
```bash
./export-onnx.py
```

It will generate a file `./m.onnx`

```bash
 ls -lh m.onnx
-rw-r--r-- 1 kuangfangjun root 627K Mar 30 11:13 m.onnx
```

## 3. Test the onnx model

```bash
wget https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/lei-jun-test.wav
./test-onnx.py  --model ./m.onnx --wav ./lei-jun-test.wav
```

## 4. Convert the onnx model to RKNN format

We assume you have installed rknn toolkit 2.1
```bash
./export-rknn.py --in-model ./m.onnx --out-model m.rknn  --target-platform rk3588
```

It will generate a file `./m.rknn`

```bash
ls -lh m.rknn
-rw-r--r-- 1 kuangfangjun root 2.2M Mar 30 11:19 m.rknn
```
