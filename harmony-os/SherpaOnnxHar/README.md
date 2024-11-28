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
