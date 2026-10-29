#!/usr/bin/env python3
# Copyright (c)  2026  Silvio Tomatis
"""
Run the exported Nemotron-3-Diarization ONNX model on a wave file.

It implements, with numpy, the Arrival-Order Speaker Cache (AOSC) and FIFO
queue of the streaming Sortformer, following
transformers.models.nemotron3_diarization.Nemotron3DiarizationSpeakerCache.

Usage:

  ./test_onnx.py --model ./model.onnx --wav ./test.wav

If --reference is given, it also runs the PyTorch model from transformers and
reports the largest difference of the speaker probabilities.
"""

import argparse
import math
from typing import Dict, List, Tuple

import librosa
import numpy as np
import onnxruntime as ort
import soundfile as sf


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--wav", type=str, required=True)
    parser.add_argument(
        "--num-samples",
        type=int,
        default=0,
        help="Use this many samples for padding/chunk boundary checks (0: all)",
    )
    parser.add_argument(
        "--streaming",
        action="store_true",
        help="Use the low-latency streaming geometry (chunk 9, right context 4)",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default="f667ed73aee57d40cc39428eb768b4fd87a0a29e",
        help="Hugging Face reference revision (ignored for a local directory)",
    )
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument(
        "--max-feature-diff",
        type=float,
        default=1e-3,
        help="Fail if the frontend reference comparison exceeds this tolerance",
    )
    parser.add_argument(
        "--max-prob-diff",
        type=float,
        default=1e-4,
        help="Fail if the FP32 reference comparison exceeds this tolerance",
    )
    parser.add_argument(
        "--reference",
        type=str,
        default="",
        help="Hugging Face model ID or directory of the PyTorch model to "
        "compare with",
    )
    return parser.parse_args()


class OnnxModel:
    def __init__(self, filename: str):
        session_opts = ort.SessionOptions()
        session_opts.inter_op_num_threads = 1
        session_opts.intra_op_num_threads = 4
        self.model = ort.InferenceSession(
            filename, sess_options=session_opts, providers=["CPUExecutionProvider"]
        )
        meta = self.model.get_modelmeta().custom_metadata_map
        self.meta = meta

        def i(key):
            return int(meta[key])

        def f(key):
            return float(meta[key])

        self.sample_rate = i("sample_rate")
        self.n_fft = i("n_fft")
        self.win_length = i("win_length")
        self.hop_length = i("hop_length")
        self.num_mel_bins = i("num_mel_bins")
        self.preemphasis = f("preemphasis")

        self.num_speakers = i("num_speakers")
        self.subsampling_factor = i("subsampling_factor")
        self.hidden_size = i("hidden_size")

        self.chunk_length = i("chunk_length")
        self.is_streaming = False
        self.chunk_right_context = i("chunk_right_context")
        self.fifo_length = i("fifo_length")
        self.speaker_cache_update_period = i("speaker_cache_update_period")

        self.speaker_cache_length = i("speaker_cache_length")
        self.num_silence_frames = i("speaker_cache_silence_frames_per_speaker")
        self.prediction_score_threshold = f("prediction_score_threshold")
        self.latest_frames_score_boost = f("latest_frames_score_boost")
        self.min_positive_scores_rate = f("min_positive_scores_rate")
        self.strong_boost_rate = f("strong_boost_rate")
        self.weak_boost_rate = f("weak_boost_rate")

        self.silence_embeds = np.array(
            [float(x) for x in meta["silence_embeds"].split(",")], dtype=np.float32
        )
        assert self.silence_embeds.shape == (self.hidden_size,)

    def use_streaming_geometry(self):
        self.is_streaming = True
        self.chunk_length = 9
        self.chunk_right_context = 4
        self.fifo_length = int(self.meta["streaming_fifo_length"])
        self.speaker_cache_update_period = int(
            self.meta["streaming_speaker_cache_update_period"]
        )

    def run(
        self, features: np.ndarray, cached_embeds: np.ndarray, num_frames: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        probs, chunk_embeds = self.model.run(
            ["probs", "chunk_embeds"],
            {
                "features": features[None],
                "cached_embeds": cached_embeds[None],
                "num_frames": np.array(num_frames, dtype=np.int64),
            },
        )
        return probs[0], chunk_embeds[0]


def compute_features(samples: np.ndarray, m: OnnxModel) -> np.ndarray:
    """NeMo log-mel features without normalization: (num_frames, num_mel_bins)

    num_frames is len(samples) // hop_length.
    """
    x = np.concatenate(
        [samples[:1], samples[1:] - m.preemphasis * samples[:-1]]
    ).astype(np.float32)
    pad = m.n_fft // 2
    x = np.pad(x, (pad, pad))
    num_frames = len(samples) // m.hop_length

    window = np.hanning(m.win_length).astype(np.float32)  # symmetric
    left = (m.n_fft - m.win_length) // 2
    window = np.pad(window, (left, m.n_fft - m.win_length - left))

    frames = np.lib.stride_tricks.sliding_window_view(x, m.n_fft)[:: m.hop_length][
        :num_frames
    ]
    power = np.abs(np.fft.rfft(frames * window, axis=-1)) ** 2

    mel = librosa.filters.mel(
        sr=m.sample_rate,
        n_fft=m.n_fft,
        n_mels=m.num_mel_bins,
        fmin=0.0,
        fmax=m.sample_rate / 2,
        norm="slaney",
    )
    return np.log(power @ mel.T + 2**-24).astype(np.float32)


class SpeakerCache:
    def __init__(self, m: OnnxModel):
        self.m = m
        budget = m.speaker_cache_length // m.num_speakers - m.num_silence_frames
        self.min_positive_scores = math.floor(budget * m.min_positive_scores_rate)
        self.num_strong_boosted = math.floor(budget * m.strong_boost_rate)
        self.num_weak_boosted = math.floor(budget * m.weak_boost_rate)

        self.cache_embeds = np.zeros((0, m.hidden_size), dtype=np.float32)
        self.cache_probs = np.zeros((0, m.num_speakers), dtype=np.float32)
        self.fifo = np.zeros((0, m.hidden_size), dtype=np.float32)
        self.is_compressed = False

    def get_embeds(self) -> np.ndarray:
        return np.concatenate([self.cache_embeds, self.fifo], axis=0)

    def update(self, chunk_embeds: np.ndarray, probs: np.ndarray, num_chunk_frames):
        """
        Args:
          chunk_embeds: embeddings of the chunk and its right context
          probs: (num_input_frames * subsampling_factor, num_speakers)
          num_chunk_frames: number of chunk frames, without right context
        """
        m = self.m
        num_cache_frames = self.cache_embeds.shape[0]
        # average over the subsampling_factor 10 ms rows of an encoder frame
        probs = probs.reshape(-1, m.subsampling_factor, m.num_speakers).mean(axis=1)

        fifo = np.concatenate([self.fifo, chunk_embeds[:num_chunk_frames]], axis=0)
        n = fifo.shape[0]
        if n <= m.fifo_length:
            self.fifo = fifo
            return

        num_popped = min(max(m.speaker_cache_update_period, n - m.fifo_length), n)
        fifo_probs = probs[num_cache_frames : num_cache_frames + n]

        stored_probs = (
            self.cache_probs if self.is_compressed else probs[:num_cache_frames]
        )
        cache_embeds = np.concatenate([self.cache_embeds, fifo[:num_popped]], axis=0)
        cache_probs = np.concatenate([stored_probs, fifo_probs[:num_popped]], axis=0)
        self.fifo = fifo[num_popped:]

        if cache_embeds.shape[0] > m.speaker_cache_length:
            cache_embeds, cache_probs = self.compress(cache_embeds, cache_probs)
            self.is_compressed = True

        self.cache_embeds = cache_embeds
        self.cache_probs = cache_probs

    def get_scores(self, probs: np.ndarray) -> np.ndarray:
        threshold = self.m.prediction_score_threshold
        with np.errstate(divide="ignore"):
            log_probs = np.log(np.maximum(probs, threshold))
            log_complements = np.log(np.maximum(1.0 - probs, threshold))
        scores = (
            log_probs
            - log_complements
            + log_complements.sum(axis=-1, keepdims=True)
            - math.log(0.5)
        )
        is_speech = probs > 0.5
        scores[~is_speech] = -np.inf
        is_positive = scores > 0
        has_enough = is_positive.sum(axis=0, keepdims=True) >= self.min_positive_scores
        scores[~is_positive & is_speech & has_enough] = -np.inf
        return scores

    @staticmethod
    def topk(column: np.ndarray, k: int) -> np.ndarray:
        # ties go to the lower index
        return np.argsort(-column, kind="stable")[:k]

    def compress(self, embeds: np.ndarray, probs: np.ndarray):
        m = self.m
        num_frames, num_speakers = probs.shape
        scores = self.get_scores(probs)
        scores[m.speaker_cache_length :] += m.latest_frames_score_boost

        for num_boosted, boost in [
            (self.num_strong_boosted, -2.0 * math.log(0.5)),
            (self.num_weak_boosted, -math.log(0.5)),
        ]:
            for s in range(num_speakers):
                idx = self.topk(scores[:, s], num_boosted)
                scores[idx, s] += boost

        num_scored = num_frames + m.num_silence_frames
        scores = np.concatenate(
            [scores, np.full((m.num_silence_frames, num_speakers), np.inf)], axis=0
        )
        flat = scores.T.reshape(-1)  # speaker-major
        picked = self.topk(flat, m.speaker_cache_length)
        sentinel = num_scored * num_speakers
        picked = np.where(flat[picked] == -np.inf, sentinel, picked)
        picked = np.sort(picked)
        frames = np.where(
            picked == sentinel, num_frames, np.minimum(picked % num_scored, num_frames)
        )

        embeds = np.concatenate([embeds, m.silence_embeds[None]], axis=0)
        probs = np.concatenate([probs, np.zeros((1, num_speakers), probs.dtype)])
        return embeds[frames], probs[frames]


def diarize(m: OnnxModel, features: np.ndarray) -> np.ndarray:
    """Returns speaker probabilities, (num_frames, num_speakers), 10 ms each."""
    sf_ = m.subsampling_factor
    num_frames = features.shape[0]
    if num_frames == 0:
        return np.zeros((0, m.num_speakers), dtype=np.float32)
    # The centered STFT has one more frame than the valid feature length.
    # The processor zeroes it, but its embedding still reaches the output
    # convolution when num_frames is a multiple of the subsampling factor.
    num_embeds = (num_frames + sf_ - int(m.is_streaming)) // sf_
    pad = num_embeds * sf_ - num_frames
    features = np.pad(features, ((0, pad), (0, 0)))

    cache = SpeakerCache(m)
    ans = []
    for start in range(0, num_embeds, m.chunk_length):
        end = min(start + m.chunk_length, num_embeds)
        num_chunk_frames = end - start
        stop = min(end + m.chunk_right_context, num_embeds)

        cached = cache.get_embeds()
        probs, chunk_embeds = m.run(
            features[start * sf_ : stop * sf_],
            cached,
            min(stop * sf_, num_frames) - start * sf_,
        )
        cache.update(chunk_embeds, probs, num_chunk_frames)

        c = cached.shape[0]
        ans.append(probs[c * sf_ : (c + num_chunk_frames) * sf_])

    return np.concatenate(ans, axis=0)[:num_frames]


def to_segments(probs: np.ndarray, threshold: float, frame_shift: float):
    active = (probs > threshold).astype(np.int8)
    zeros = np.zeros((1, active.shape[1]), dtype=np.int8)
    changes = np.diff(np.concatenate([zeros, active, zeros]), axis=0)
    segments = []
    for s in range(active.shape[1]):
        starts = np.nonzero(changes[:, s] == 1)[0]
        ends = np.nonzero(changes[:, s] == -1)[0]
        segments += [
            (b * frame_shift, e * frame_shift, s) for b, e in zip(starts, ends)
        ]
    segments.sort()
    return segments


def run_reference(
    model_id: str,
    samples: np.ndarray,
    streaming: bool,
    revision: str = "f667ed73aee57d40cc39428eb768b4fd87a0a29e",
):
    import torch
    from transformers import AutoModelForAudioFrameClassification, AutoProcessor

    processor = AutoProcessor.from_pretrained(model_id, revision=revision)
    model = AutoModelForAudioFrameClassification.from_pretrained(
        model_id, revision=revision, attn_implementation="eager", dtype=torch.float32
    )
    model.eval()

    sr = processor.feature_extractor.sampling_rate
    with torch.inference_mode():
        if not streaming:
            inputs = processor(samples, sampling_rate=sr)
            logits = model(
                input_features=inputs.input_features,
                attention_mask=inputs.attention_mask,
            ).logits
            return logits.sigmoid()[0].numpy(), inputs.input_features[0].numpy()

        processor.set_streaming_mode("low_latency")
        speaker_cache, logits = None, []

        def step(audio, first, last):
            nonlocal speaker_cache
            inputs = processor(
                audio,
                sampling_rate=sr,
                is_streaming=True,
                is_first_audio_chunk=first,
                is_last_audio_chunk=last,
            )
            out = model(**inputs, speaker_cache=speaker_cache)
            speaker_cache = out.speaker_cache
            logits.append(out.logits)

        step(samples[: processor.num_samples_first_audio_chunk], True, False)
        mel_frame_idx = processor.num_mel_frames_per_step
        start = processor.audio_chunk_start(mel_frame_idx)
        while (end := start + processor.num_samples_per_audio_chunk) <= len(samples):
            step(samples[start:end], False, False)
            mel_frame_idx += processor.num_mel_frames_per_step
            start = processor.audio_chunk_start(mel_frame_idx)
        step(samples[start:], False, True)
        return torch.cat(logits, dim=1).sigmoid()[0].numpy(), None


def main():
    args = get_args()
    m = OnnxModel(args.model)
    if args.streaming:
        m.use_streaming_geometry()

    samples, sample_rate = sf.read(args.wav, dtype="float32", always_2d=True)
    samples = samples[:, 0]
    if sample_rate != m.sample_rate:
        samples = librosa.resample(
            samples, orig_sr=sample_rate, target_sr=m.sample_rate
        )

    if args.num_samples > 0:
        samples = samples[: args.num_samples]

    features = compute_features(samples, m)
    if len(features) == 0:
        print("No complete 10 ms frames to process")
        return
    probs = diarize(m, features)

    frame_shift = m.hop_length / m.sample_rate
    for b, e, s in to_segments(probs, args.threshold, frame_shift):
        print(f"{b:8.2f} -- {e:8.2f} speaker_{s:02d}")

    if args.reference:
        ref_probs, ref_features = run_reference(
            args.reference, samples, args.streaming, args.revision
        )
        if len(ref_probs) not in (len(probs), len(probs) + 1):
            raise RuntimeError(
                f"Unexpected frame counts: ONNX {len(probs)}, reference {len(ref_probs)}"
            )
        if ref_features is not None:
            n = features.shape[0]
            max_feature_diff = np.abs(ref_features[:n] - features).max()
            print("max feature diff:", max_feature_diff)
            if (
                not np.isfinite(max_feature_diff)
                or max_feature_diff > args.max_feature_diff
            ):
                raise RuntimeError(
                    f"Feature difference {max_feature_diff} exceeds {args.max_feature_diff}"
                )
        n = min(len(ref_probs), len(probs))
        diff = np.abs(ref_probs[:n] - probs[:n])
        print(f"frames: onnx {len(probs)}, reference {len(ref_probs)}")
        max_diff = diff.max()
        print("max prob diff:", max_diff, "mean:", diff.mean())
        agree = (ref_probs[:n] > args.threshold) == (probs[:n] > args.threshold)
        print("decision agreement:", agree.mean())
        if not np.isfinite(max_diff) or max_diff > args.max_prob_diff:
            raise RuntimeError(
                f"Probability difference {max_diff} exceeds {args.max_prob_diff}"
            )


if __name__ == "__main__":
    main()
