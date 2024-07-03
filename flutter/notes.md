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