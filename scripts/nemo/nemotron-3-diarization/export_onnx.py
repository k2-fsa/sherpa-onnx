#!/usr/bin/env python3
# Copyright (c)  2026  Silvio Tomatis
"""
Export https://huggingface.co/nvidia/Nemotron-3-Diarization to ONNX.

The exported model is stateless. It encodes one step of the streaming
Sortformer: the speaker-cache and FIFO embeddings of the previous steps,
followed by the mel frames of the current chunk and its right context.
The Arrival-Order Speaker Cache (AOSC) and the FIFO queue are maintained by
the caller (see ./test_onnx.py and
sherpa-onnx/csrc/offline-speaker-diarization-sortformer-impl.h).

Inputs:
  - features: (N, T, 128), log-mel frames, T must be a multiple of 8
  - cached_embeds: (N, C, 512), speaker cache + FIFO embeddings; C may be 0
  - num_frames: scalar int64, valid mel frames in features (before padding)

Outputs:
  - probs: (N, (C + T/8) * 8, 8), sigmoid speaker activity, one row per 10 ms
  - chunk_embeds: (N, T/8, 512), embeddings of the chunk frames, to be pushed
    to the FIFO queue
"""

import argparse
from typing import Dict

import onnx
import torch
from onnxruntime.quantization import QuantType, quantize_dynamic
from transformers import AutoModelForAudioFrameClassification, AutoProcessor


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-id",
        type=str,
        default="nvidia/Nemotron-3-Diarization",
        help="Hugging Face model ID or local directory",
    )
    parser.add_argument("--opset", type=int, default=17)
    return parser.parse_args()


def add_meta_data(filename: str, meta_data: Dict[str, str]):
    """Add meta data to an ONNX model. It is changed in-place."""
    model = onnx.load(filename)

    while len(model.metadata_props):
        model.metadata_props.pop()

    for key, value in meta_data.items():
        meta = model.metadata_props.add()
        meta.key = key
        meta.value = str(value)

    onnx.save(model, filename)


class OnnxModel(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model.model
        self.classifier = model.classifier
        self.projection = model.model.audio_tower.embedder.projection
        self.subsampling_factor = model.config.audio_config.subsampling_factor

    def forward(
        self,
        features: torch.Tensor,
        cached_embeds: torch.Tensor,
        num_frames: torch.Tensor,
    ):
        """
        Args:
          features: (N, T, num_mel_bins), T % subsampling_factor == 0
          cached_embeds: (N, C, hidden_size)
          num_frames: scalar, number of valid mel frames in features
        Returns:
          probs: (N, (C + T / subsampling_factor) * subsampling_factor,
                  num_speakers)
          chunk_embeds: (N, T / subsampling_factor, hidden_size)
        """
        n = features.shape[0]
        # Feature stacking. The caller zero-pads the last group, like
        # Nemotron3DiarizationFeatureStacking does.
        stacked = features.reshape(n, -1, features.shape[2] * self.subsampling_factor)
        chunk_embeds = self.projection(stacked)

        x = torch.cat([cached_embeds, chunk_embeds], dim=1)

        # Positions restart at every step
        position_ids = torch.arange(x.shape[1], device=x.device).unsqueeze(0)
        valid_embeds = (
            num_frames + self.subsampling_factor - 1
        ) // self.subsampling_factor
        attention_mask = position_ids < cached_embeds.shape[1] + valid_embeds
        hidden = self.model(
            inputs_embeds=x,
            position_ids=position_ids,
            attention_mask=attention_mask,
        )
        logits = self.classifier(hidden.last_hidden_state)
        return logits.sigmoid(), chunk_embeds


def floats_to_str(t: torch.Tensor) -> str:
    return ",".join(f"{v:.9g}" for v in t.reshape(-1).tolist())


@torch.no_grad()
def main():
    args = get_args()
    print(vars(args))

    model = AutoModelForAudioFrameClassification.from_pretrained(
        args.model_id, attn_implementation="eager", dtype=torch.float32
    )
    model.eval()
    processor = AutoProcessor.from_pretrained(args.model_id)
    fe = processor.feature_extractor

    config = model.config
    audio = config.audio_config
    head = config.head_config
    streaming = config.streaming_config

    num_mel_bins = audio.num_mel_bins
    hidden_size = audio.hidden_size
    subsampling_factor = audio.subsampling_factor

    onnx_model = OnnxModel(model)
    onnx_model.eval()

    features = torch.randn(1, 13 * subsampling_factor, num_mel_bins)
    cached_embeds = torch.randn(1, 7, hidden_size)
    num_frames = torch.tensor(features.shape[1], dtype=torch.int64)

    filename = "model.onnx"
    torch.onnx.export(
        onnx_model,
        (features, cached_embeds, num_frames),
        filename,
        input_names=["features", "cached_embeds", "num_frames"],
        output_names=["probs", "chunk_embeds"],
        dynamic_axes={
            "features": {0: "N", 1: "T"},
            "cached_embeds": {0: "N", 1: "C"},
            "probs": {0: "N", 1: "T_out"},
            "chunk_embeds": {0: "N", 1: "T_chunk"},
        },
        opset_version=args.opset,
        dynamo=False,
    )

    meta_data = {
        "model_type": "nemotron3_diarization",
        "version": 2,
        "model_author": "NVIDIA",
        "url": "https://huggingface.co/nvidia/Nemotron-3-Diarization",
        "license": "https://huggingface.co/nvidia/Nemotron-3-Diarization",
        "comment": "Streaming Sortformer with Arrival-Order Speaker Cache",
        # frontend
        "sample_rate": fe.sampling_rate,
        "n_fft": fe.n_fft,
        "win_length": fe.win_length,
        "hop_length": fe.hop_length,
        "num_mel_bins": num_mel_bins,
        "preemphasis": fe.preemphasis,
        # network
        "num_speakers": head.num_speakers,
        "subsampling_factor": subsampling_factor,
        "hidden_size": hidden_size,
        "max_position_embeddings": audio.max_position_embeddings,
        # offline streaming geometry, in encoder frames (80 ms)
        "chunk_length": config.chunk_length,
        "chunk_right_context": config.chunk_right_context,
        "fifo_length": config.fifo_length,
        "speaker_cache_update_period": config.speaker_cache_update_period,
        # low-latency streaming geometry
        "streaming_fifo_length": streaming.fifo_length,
        "streaming_speaker_cache_update_period": streaming.speaker_cache_update_period,
        # speaker cache policy
        "speaker_cache_length": streaming.speaker_cache_length,
        "speaker_cache_silence_frames_per_speaker": streaming.speaker_cache_silence_frames_per_speaker,
        "prediction_score_threshold": streaming.prediction_score_threshold,
        "latest_frames_score_boost": streaming.latest_frames_score_boost,
        "min_positive_scores_rate": streaming.min_positive_scores_rate,
        "strong_boost_rate": streaming.strong_boost_rate,
        "weak_boost_rate": streaming.weak_boost_rate,
        "silence_embeds": floats_to_str(model.silence_embeds),
    }
    print(meta_data)
    add_meta_data(filename, meta_data)

    filename_int8 = "model.int8.onnx"
    quantize_dynamic(
        model_input=filename,
        model_output=filename_int8,
        op_types_to_quantize=["MatMul"],
        weight_type=QuantType.QInt8,
    )


if __name__ == "__main__":
    torch.manual_seed(20261007)
    main()
