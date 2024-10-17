#!/usr/bin/env python3
# Copyright      2024  Xiaomi Corp.        (authors: Fangjun Kuang)

"""
Please refer to
https://github.com/k2-fsa/sherpa-onnx/blob/master/.github/workflows/speaker-diarization.yaml
for usages.
"""

"""
1. Go to https://huggingface.co/hbredin/wespeaker-voxceleb-resnet34-LM/tree/main
wget https://huggingface.co/hbredin/wespeaker-voxceleb-resnet34-LM/resolve/main/speaker-embedding.onnx

2. Change line 166 of pyannote/audio/pipelines/speaker_diarization.py

```
            #  self._embedding = PretrainedSpeakerEmbedding(
            #      self.embedding, use_auth_token=use_auth_token
            #  )
            self._embedding = embedding
```
"""

import argparse
from pathlib import Path

import torch
from pyannote.audio import Model
from pyannote.audio.pipelines import SpeakerDiarization as SpeakerDiarizationPipeline
from pyannote.audio.pipelines.speaker_verification import (
    ONNXWeSpeakerPretrainedSpeakerEmbedding,
)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--wav", type=str, required=True, help="Path to test.wav")

    return parser.parse_args()


def build_pipeline():
    embedding_filename = "./speaker-embedding.onnx"
    if Path(embedding_filename).is_file():
        # You need to modify line 166
        # of pyannote/audio/pipelines/speaker_diarization.py
        # Please see the comments at the start of this script for details
        embedding = ONNXWeSpeakerPretrainedSpeakerEmbedding(embedding_filename)
    else:
        embedding = "hbredin/wespeaker-voxceleb-resnet34-LM"

    pt_filename = "./pytorch_model.bin"
    segmentation = Model.from_pretrained(pt_filename)
    segmentation.eval()

    pipeline = SpeakerDiarizationPipeline(
        segmentation=segmentation,
        embedding=embedding,
        embedding_exclude_overlap=True,
    )

    params = {
        "clustering": {
            "method": "centroid",
            "min_cluster_size": 12,
            "threshold": 0.7045654963945799,
        },
        "segmentation": {"min_duration_off": 0.5},
    }

    pipeline.instantiate(params)
    return pipeline


@torch.no_grad()
def main():
    args = get_args()
    assert Path(args.wav).is_file(), args.wav
    pipeline = build_pipeline()
    print(pipeline)
    t = pipeline(args.wav)
    print(type(t))
    print(t)


if __name__ == "__main__":
    main()
