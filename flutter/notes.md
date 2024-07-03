# Introduction

This file keeps some notes about how packages in this directory
are created.

1. Create `sherpa_onnx`.

```bash
flutter create --template plugin sherpa_onnx
```

2. Create `sherpa_onnx_macos`

```bash
flutter create --template plugin_ffi --platforms macos sherpa_onnx_macos
```

3. Create `sherpa_onnx_linux

```bash
flutter create --template plugin_ffi --platforms linux sherpa_onnx_linux
```

4. Create `sherpa_onnx_windows

```bash
flutter create --template plugin_ffi --platforms linux sherpa_onnx_windows
```

5. Create `sherpa_onnx_android

```bash
flutter create --template plugin_ffi --platforms android --org com.k2fsa.sherpa.onnx sherpa_onnx_android
```
