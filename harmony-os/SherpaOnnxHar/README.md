# Introduction

How to build `sherpa_onnx.har` from the command line:

```bash
git clone https://github.com/k2-fsa/sherpa-onnx
cd sherpa-onnx
./build-ohos-arm64-v8a.sh
./build-ohos-x86-64.sh

cd harmony-os/SherpaOnnxHar

hvigorw clean --no-daemon

hvigorw --mode module -p product=default -p module=sherpa_onnx@default assembleHar --analyze=normal --parallel --incremental --no-daemon

ls -lh ./sherpa_onnx/build/default/outputs/default/sherpa_onnx.har
```

Pre-built `har` packages can be found at
<https://huggingface.co/csukuangfj/sherpa-onnx-harmony-os/tree/main/har>

You can also download it using
```
wget https://ohpm.openharmony.cn/ohpm/sherpa_onnx/-/sherpa_onnx-1.10.33.har

# Please replace the version 1.10.33 if needed.
```

You can also use
```
ohpm install sherpa_onnx
```
to install it.

See also
<https://ohpm.openharmony.cn/#/cn/detail/sherpa_onnx>
