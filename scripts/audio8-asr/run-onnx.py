#!/usr/bin/env python3
"""Run the exported Audio8-ASR ONNX graphs and print the transcript.

This is the small reference implementation used to sanity check the export. It
only needs `onnxruntime`, `kaldi-native-fbank`, `numpy`, `librosa` and
`tokenizers` — the same feature extraction / tokenizer stack that the C++
runtime uses, not `transformers`.

Usage::

    python3 run-onnx.py wavs/0.wav [more.wav ...]
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import onnxruntime as ort

ROOT = Path(__file__).resolve().parent
FRAMES = 3000                       # 30 s of 10 ms mel frames
MAX_SAMPLES = FRAMES * 160
MEL_DIM = 128

# tokens of the arkasr prompt (see scripts/audio8-asr/README.md)
USER, BEGIN_OF_AUDIO, AUDIO = 151647, 151648, 151646
END_OF_AUDIO, ASSISTANT = 151649, 151650
EOS_ID, PAD_ID = 151645, 151643
PROMPT_TEXT = "Please transcribe this audio."


# --------------------------------------------------------------------------- #
# audio
# --------------------------------------------------------------------------- #
def read_wav(path: str) -> np.ndarray:
    """Return mono float32 samples at 16 kHz, truncated to the 30 s window."""
    import librosa

    samples, _ = librosa.load(path, sr=16000, mono=True)
    if len(samples) > MAX_SAMPLES:
        print(f"[warn] {path}: truncated to 30 s")
        samples = samples[:MAX_SAMPLES]
    return samples.astype(np.float32)


def log_mel(samples: np.ndarray) -> np.ndarray:
    """Whisper log-mel, [1, 128, 3000], zero padded (same as the C++ runtime)."""
    import kaldi_native_fbank as knf

    opts = knf.WhisperFeatureOptions()
    opts.dim = MEL_DIM
    # the C++ runtime uses the defaults of sherpa-onnx's FeatureExtractorConfig:
    # dither off, and the whisper code path ignores the other frame options
    opts.frame_opts.dither = 0.0
    fbank = knf.OnlineWhisperFbank(opts)
    # sherpa-onnx prepends half a frame shift of silence so that the window
    # midpoints follow the centered STFT convention of the reference extractor,
    # and then keeps floor(num_samples / 160) frames (see OfflineStream with
    # WhisperTag + align_to_stft_center).
    half_shift = np.zeros(80, dtype=np.float32)
    padded = np.concatenate([half_shift, samples])
    fbank.accept_waveform(16000, padded)
    fbank.input_finished()

    num_frames = min(fbank.num_frames_ready, len(samples) // 160, FRAMES)
    mel = np.stack([fbank.get_frame(i) for i in range(num_frames)]).astype(np.float32)

    # whisper normalisation: log10(max(x, 1e-10)), clamp to max-8, (x + 4) / 4
    mel = np.log10(np.maximum(mel, 1e-10))
    mel = np.maximum(mel, mel.max() - 8.0) if mel.size else mel
    mel = (mel + 4.0) / 4.0

    if num_frames < FRAMES:
        mel = np.pad(mel, ((0, FRAMES - num_frames), (0, 0)))
    return mel.T[None]                                              # [1, 128, 3000]


def num_mel_frames(samples: np.ndarray) -> int:
    return max(1, len(samples) // 160)


def encoder_rows(mel_frames: int) -> int:
    """Rows the encoder produces: 13 per 100 frames + partial chunk."""
    def after_cnn(x: int) -> int:
        x = (x - 1) // 2 + 1
        x = (x - 1) // 2 + 1
        return (x - 1) // 2 + 1

    full, rem = divmod(mel_frames, 100)
    return 13 * full + (after_cnn(rem) if rem else 0)


def audio_tokens(mel_frames: int) -> int:
    return max(1, ((mel_frames + 1) // 2) // 4)


def adaptive_avg_pool(x: np.ndarray, out_len: int) -> np.ndarray:
    n = x.shape[0]
    if n == out_len:
        return x
    out = np.empty((out_len, x.shape[1]), dtype=np.float32)
    for i in range(out_len):
        start = math.floor(i * n / out_len)
        end = math.ceil((i + 1) * n / out_len)
        out[i] = x[start:end].mean(axis=0)
    return out


# --------------------------------------------------------------------------- #
# text
# --------------------------------------------------------------------------- #
class Tokenizer:
    def __init__(self, tokenizer_dir: Path):
        from tokenizers import Tokenizer as HFTokenizer

        self.tok = HFTokenizer.from_file(str(tokenizer_dir / "tokenizer.json"))

    def encode(self, text: str) -> list[int]:
        return self.tok.encode(text, add_special_tokens=False).ids

    def decode(self, ids: list[int]) -> str:
        return self.tok.decode(ids, skip_special_tokens=True).strip()


# --------------------------------------------------------------------------- #
# runtime
# --------------------------------------------------------------------------- #
class Audio8Asr:
    def __init__(self, exported: Path, tokenizer_dir: Path, threads: int = 4):
        so = ort.SessionOptions()
        so.intra_op_num_threads = threads
        so.inter_op_num_threads = 1
        self.encoder = ort.InferenceSession(str(exported / "encoder.onnx"), so,
                                            providers=["CPUExecutionProvider"])
        self.decoder = ort.InferenceSession(str(exported / "decoder.onnx"), so,
                                            providers=["CPUExecutionProvider"])
        self.meta = json.loads((exported / "metadata.json").read_text())
        self.tok = Tokenizer(tokenizer_dir)
        self.max_total_len = int(self.meta["max_total_len"])

    def prompt_ids(self, n_audio: int) -> list[int]:
        return ([USER, BEGIN_OF_AUDIO] + [AUDIO] * n_audio + [END_OF_AUDIO]
                + self.tok.encode(PROMPT_TEXT) + [ASSISTANT])

    def audio_features(self, samples: np.ndarray) -> np.ndarray:
        mel_frames = min(num_mel_frames(samples), FRAMES)
        n_tokens = audio_tokens(mel_frames)
        keep = min(encoder_rows(mel_frames), FRAMES // 100 * 13)
        hidden = self.encoder.run(
            None,
            {"audios": log_mel(samples),
             "num_rows": np.array([keep], dtype=np.int64)})[0][0]
        return adaptive_avg_pool(hidden[:keep], n_tokens)[None]      # [1, N, 1024]

    def transcribe(self, path: str, max_new_tokens: int = 128) -> str:
        samples = read_wav(path)
        n_audio = audio_tokens(min(num_mel_frames(samples), FRAMES))
        audio = self.audio_features(samples)
        ids = self.prompt_ids(n_audio)

        if len(ids) + 1 > self.max_total_len:
            raise SystemExit(
                f"prompt ({len(ids)}) does not fit the {self.max_total_len}-slot "
                "cache; use shorter audio or re-export with a larger --max-total-len")
        max_new_tokens = min(max_new_tokens, self.max_total_len - len(ids) - 1)

        n_layers = int(self.meta["num_layers"])
        n_kv = int(self.meta["num_kv_heads"])
        head_dim = int(self.meta["head_dim"])
        caches = []
        for _ in range(n_layers):
            caches.append(np.zeros((1, n_kv, self.max_total_len, head_dim),
                                   dtype=np.float32))
            caches.append(np.zeros((1, n_kv, self.max_total_len, head_dim),
                                   dtype=np.float32))

        attn = np.zeros((1, self.max_total_len), dtype=np.int64)
        attn[0, : len(ids)] = 1
        feeds = {"input_ids": np.array([ids], dtype=np.int64), "audio_hidden": audio,
                 "attention_mask": attn,
                 "cache_position": np.arange(len(ids), dtype=np.int64)}
        for i in range(n_layers):
            feeds[f"cache_key_{i}"] = caches[2 * i]
            feeds[f"cache_value_{i}"] = caches[2 * i + 1]
        outs = self.decoder.run(None, feeds)
        logits = outs[0][0, -1, :]
        for i in range(n_layers):
            caches[2 * i][:, :, : len(ids), :] = outs[1 + 2 * i]
            caches[2 * i + 1][:, :, : len(ids), :] = outs[2 + 2 * i]

        generated: list[int] = []
        pos = len(ids)
        for _ in range(max_new_tokens):
            nxt = int(np.argmax(logits))
            if nxt == EOS_ID or nxt == PAD_ID:
                break
            generated.append(nxt)
            attn = np.zeros((1, self.max_total_len), dtype=np.int64)
            attn[0, : pos + 1] = 1
            feeds = {"input_ids": np.array([[nxt]], dtype=np.int64),
                     "audio_hidden": audio, "attention_mask": attn,
                     "cache_position": np.array([pos], dtype=np.int64)}
            for i in range(n_layers):
                feeds[f"cache_key_{i}"] = caches[2 * i]
                feeds[f"cache_value_{i}"] = caches[2 * i + 1]
            outs = self.decoder.run(None, feeds)
            logits = outs[0][0, -1, :]
            for i in range(n_layers):
                caches[2 * i][:, :, pos: pos + 1, :] = outs[1 + 2 * i]
                caches[2 * i + 1][:, :, pos: pos + 1, :] = outs[2 + 2 * i]
            pos += 1

        return self.tok.decode(generated)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("wavs", nargs="+")
    ap.add_argument("--onnx-dir", default=str(ROOT / "exported"))
    ap.add_argument("--tokenizer", default=None,
                    help="Tokenizer directory; defaults to --model-dir")
    ap.add_argument("--model-dir", default=str(ROOT / "Audio8-ASR-0.1B"))
    ap.add_argument("--num-threads", type=int, default=4)
    ap.add_argument("--max-new-tokens", type=int, default=128)
    args = ap.parse_args()

    tokenizer_dir = Path(args.tokenizer or args.model_dir)
    asr = Audio8Asr(Path(args.onnx_dir), tokenizer_dir, args.num_threads)
    for wav in args.wavs:
        print(f"{wav}: {asr.transcribe(wav, args.max_new_tokens)}")


if __name__ == "__main__":
    main()
