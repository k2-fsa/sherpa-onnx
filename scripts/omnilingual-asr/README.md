# Introduction

This folder contains script to export
https://github.com/facebookresearch/omnilingual-asr
to sherpa-onnx

See
https://github.com/k2-fsa/sherpa-onnx/blob/master/.github/workflows/export-omnilingual-asr-to-onnx.yaml
for usage.

```
num_frames = round(num_samples / 318 - 1.5)
num_samples = round(318 * num_frames + 477)

or
num_frames = round(num_samples / 320)

```

20ms per frame



