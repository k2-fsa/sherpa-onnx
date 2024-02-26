# Introduction

This directory contains script for exporting speaker verification models
from [NeMo](https://github.com/NVIDIA/NeMo/) to onnx
so that you can use them in `sherpa-onnx`.

Specifically, the following 4 models are exported to `sherpa-onnx`
from
[this page](https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/main/asr/speaker_recognition/results.html#speaker-recognition-models):

  - [titanet_large](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/nemo/models/titanet_large),
  - [titanet_small](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/nemo/models/titanet_small)
  - [speakerverification_speakernet](https://ngc.nvidia.com/catalog/models/nvidia:nemo:speakerverification_speakernet)
  - [ecapa_tdnn](https://ngc.nvidia.com/catalog/models/nvidia:nemo:ecapa_tdnn)
