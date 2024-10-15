# Introduction

Please refer to
https://github.com/k2-fsa/sherpa-onnx/releases/tag/speaker-segmentation-models
to download a speaker segmentation model
and
refer to
https://github.com/k2-fsa/sherpa-onnx/releases/tag/speaker-recongition-models
to download a speaker embedding extraction model.

Remember to rename the downloaded files.

The following is an example.

```bash
cd wasm/speaker-diarization/assets/

curl -SL -O https://github.com/k2-fsa/sherpa-onnx/releases/download/speaker-segmentation-models/sherpa-onnx-pyannote-segmentation-3-0.tar.bz2
tar xvf sherpa-onnx-pyannote-segmentation-3-0.tar.bz2
rm sherpa-onnx-pyannote-segmentation-3-0.tar.bz2
cp sherpa-onnx-pyannote-segmentation-3-0/model.onnx ./segmentation.onnx
rm -rf sherpa-onnx-pyannote-segmentation-3-0

curl -SL -O https://github.com/k2-fsa/sherpa-onnx/releases/download/speaker-recongition-models/3dspeaker_speech_eres2net_base_sv_zh-cn_3dspeaker_16k.onnx
mv 3dspeaker_speech_eres2net_base_sv_zh-cn_3dspeaker_16k.onnx ./embedding.onnx
```
