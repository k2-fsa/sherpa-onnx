#!/usr/bin/env python3

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

import torch
from pathlib import Path
from pyannote.audio import Model
from pyannote.audio.pipelines import SpeakerDiarization as SpeakerDiarizationPipeline
from pyannote.audio.pipelines.speaker_verification import (
    PretrainedSpeakerEmbedding,
    ONNXWeSpeakerPretrainedSpeakerEmbedding,
)


def build_pipeline():
    embedding_filename = "./speaker-embedding.onnx"
    if Path(embedding_filename).is_file():
        # You need to modify line 166
        # of pyannote/audio/pipelines/speaker_diarization.py
        # Please see the comments at the begin of this script for details
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
    pipeline = build_pipeline()
    print(pipeline)
    #  t = pipeline("./lei-jun-test.wav")
    #  t = pipeline("./test_16k.wav")
    #  t = pipeline("./2speakers_example.wav")
    #  t = pipeline("./data_afjiv.wav")
    t = pipeline("./fc-2speakers.wav")
    #  t = pipeline("./ML16091-Audio.wav")
    print(type(t))
    print(t)


if __name__ == "__main__":
    main()
