#!/usr/bin/env python3

import torch
from pyannote.audio import Model
from pyannote.audio.pipelines import (
    VoiceActivityDetection as VoiceActivityDetectionPipeline,
)


@torch.no_grad()
def main():
    # Please download it from
    # https://huggingface.co/csukuangfj/pyannote-models/tree/main/segmentation-3.0
    pt_filename = "./pytorch_model.bin"
    model = Model.from_pretrained(pt_filename)
    model.eval()

    pipeline = VoiceActivityDetectionPipeline(segmentation=model)

    # https://huggingface.co/pyannote/voice-activity-detection/blob/main/config.yaml
    # https://github.com/pyannote/pyannote-audio/issues/1215
    initial_params = {
        "min_duration_on": 0.0,
        "min_duration_off": 0.0,
    }
    pipeline.onset = 0.5
    pipeline.offset = 0.5

    pipeline.instantiate(initial_params)

    # wget https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/lei-jun-test.wav
    t = pipeline("./lei-jun-test.wav")
    print(type(t))
    print(t)


if __name__ == "__main__":
    main()
