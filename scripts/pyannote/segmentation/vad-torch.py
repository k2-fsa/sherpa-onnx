#!/usr/bin/env python3

import torch
from pyannote.audio import Model
from pyannote.audio.pipelines import (
    VoiceActivityDetection as VoiceActivityDetectionPipeline,
)


@torch.no_grad()
def main():
    pt_filename = "./pytorch_model.bin"
    model = Model.from_pretrained(pt_filename)
    model.eval()

    pipeline = VoiceActivityDetectionPipeline(segmentation=model)
    print(pipeline.segmentation.example_output)
    return

    # https://huggingface.co/pyannote/voice-activity-detection/blob/main/config.yaml
    # https://github.com/pyannote/pyannote-audio/issues/1215
    initial_params = {
        "min_duration_on": 0.0,
        "min_duration_off": 0.0,
    }
    pipeline.onset = 0.5
    pipeline.offset = 0.5

    pipeline.instantiate(initial_params)

    t = pipeline("./test_16k.wav")
    print(type(t))
    print(t)


if __name__ == "__main__":
    main()
