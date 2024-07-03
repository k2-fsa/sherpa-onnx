# Introduction

This directory contains flutter examples of `sherpa-onnx`.

# Ways to create an example
```bash
flutter create --platforms windows,macos streaming_asr
cd streaming_asr
flutter pub get

# to support a new platform, e.g., android, use

cd streaming_asr
flutter create --platforms --org com.k2fsa.sherpa.onnx android ./
```
