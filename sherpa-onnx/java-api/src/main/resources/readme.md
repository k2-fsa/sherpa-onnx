please downlaod file and put in folder
[donwload link](https://huggingface.co/csukuangfj2/sherpa-onnx-libs/tree/main/jni)

- sherpa-onnx-v1.12.7-linux-aarch64-jni.tar.bz2
- sherpa-onnx-v1.12.7-linux-x64-jni.tar.bz2
- sherpa-onnx-v1.12.7-osx-arm64-jni.tar.bz2
- sherpa-onnx-v1.12.7-osx-x86_64-jni.tar.bz2
- sherpa-onnx-v1.12.7-win-x64-jni.tar.bz2


- linux_arm64
- linux_x64
- darwin_arm64
- darwin_x64
- windows_x64


add to src/main/resources

```
.
├── native
│   ├── linux-aarch64
│   │   ├── libsherpa-onnx-jni.so
│   ├── linux-x64
│   │   ├── libsherpa-onnx-jni.so
│   ├── osx-aarch64
│   │   ├── libsherpa-onnx-jni.dylib
│   ├── osx-x64
│   │   ├── libsherpa-onnx-jni.dylib
│   ├── win-x64
│   │   ├── sherpa-onnx-jni.dll
```

