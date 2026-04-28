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

3. Create `sherpa_onnx_linux`

```bash
flutter create --template plugin_ffi --platforms linux sherpa_onnx_linux
```

4. Create `sherpa_onnx_windows`

```bash
flutter create --template plugin_ffi --platforms linux sherpa_onnx_windows
```

5. Create `sherpa_onnx_android_arm64`, `sherpa_onnx_android_armeabi`, `sherpa_onnx_android_x86`, `sherpa_onnx_android_x86_64`

```bash
flutter create --template plugin_ffi --platforms android --org com.k2fsa.sherpa.onnx.arm64 sherpa_onnx_android_arm64
flutter create --template plugin_ffi --platforms android --org com.k2fsa.sherpa.onnx.armeabi sherpa_onnx_android_armeabi
flutter create --template plugin_ffi --platforms android --org com.k2fsa.sherpa.onnx.x86 sherpa_onnx_android_x86
flutter create --template plugin_ffi --platforms android --org com.k2fsa.sherpa.onnx.x86_64 sherpa_onnx_android_x86_64
```

6. Create `sherpa_onnx_ios`

```bash
flutter create --template plugin_ffi --platforms ios sherpa_onnx_ios
```
