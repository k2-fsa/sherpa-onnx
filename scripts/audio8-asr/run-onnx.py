"""Reference ONNX runtime for the exported Audio8 graphs (Python, mirrors the C++).

Pipeline:
  wav -> Whisper log-mel (128, 3000) -> encoder.onnx -> 390 x 1024
      -> adaptive average pool to N audio tokens -> decoder.onnx (prefill + greedy)
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
from transformers import AutoProcessor

ROOT = Path(__file__).resolve().parent
EXPORTED = Path(os.environ.get("AUDIO8_ONNX_DIR", str(ROOT / "exported")))
HF_DIR = Path(os.environ.get("AUDIO8_HF_DIR", str(ROOT / "Audio8-ASR-0.1B")))

AUDIO_TOKEN_ID = 151646
USER, BOS_AUDIO, EOS_AUDIO, ASSISTANT = 151647, 151648, 151649, 151650
EOS_IDS = {151645}
PAD_ID = 151643
FRAMES = 3000
MAX_SAMPLES = FRAMES * 160        # the model is trained on 30 s windows


def audio_token_count(num_samples: int, hop: int = 160, merge: int = 4) -> int:
    mel = num_samples // hop
    return max(1, ((mel + 1) // 2) // merge)


def load_wav(path: str) -> np.ndarray:
    """Read a wav file as 16 kHz mono float32, truncated to the 30 s window."""
    try:
        import soundfile as sf

        pcm, rate = sf.read(path, dtype="float32", always_2d=True)
        pcm = pcm.mean(axis=1)
    except Exception:
        import wave

        with wave.open(path, "rb") as f:
            if f.getsampwidth() != 2:
                raise SystemExit(f"{path}: only 16-bit PCM wav files are supported")
            rate = f.getframerate()
            channels = f.getnchannels()
            pcm = np.frombuffer(f.readframes(f.getnframes()),
                                dtype=np.int16).astype(np.float32) / 32768.0
            if channels > 1:
                pcm = pcm.reshape(-1, channels).mean(axis=1)

    if rate != 16000:
        n = int(round(len(pcm) * 16000 / rate))
        pcm = np.interp(
            np.linspace(0, len(pcm) - 1, n, dtype=np.float64),
            np.arange(len(pcm)), pcm).astype(np.float32)
        print(f"[warn] {path}: resampled {rate} Hz -> 16000 Hz", flush=True)

    if len(pcm) > MAX_SAMPLES:
        print(f"[warn] {path}: truncated to {MAX_SAMPLES / 16000:.0f} s", flush=True)
        pcm = pcm[:MAX_SAMPLES]
    return pcm.astype(np.float32)


def adaptive_avg_pool(x: np.ndarray, out_len: int) -> np.ndarray:
    """x: [n, d] -> [out_len, d], same semantics as F.adaptive_avg_pool1d."""
    n = x.shape[0]
    if n == out_len:
        return x
    out = np.empty((out_len, x.shape[1]), dtype=np.float32)
    for i in range(out_len):
        start = math.floor(i * n / out_len)
        end = math.ceil((i + 1) * n / out_len)
        out[i] = x[start:end].mean(axis=0)
    return out


def after_cnn(x: int) -> int:
    """Output frames of the three stride-2 convolutions for x input frames."""
    if x <= 0:
        return 0
    x = (x - 1) // 2 + 1
    x = (x - 1) // 2 + 1
    return (x - 1) // 2 + 1


def encoder_rows(num_mel_frames: int) -> int:
    """Number of rows the encoder produces for `num_mel_frames` (13 per 100)."""
    full, rem = divmod(num_mel_frames, 100)
    return 13 * full + (after_cnn(rem) if rem else 0)


class Audio8Onnx:
    def __init__(self, exported: Path = EXPORTED, threads: int = 4):
        so = ort.SessionOptions()
        so.intra_op_num_threads = threads
        so.inter_op_num_threads = 1
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.encoder = ort.InferenceSession(str(exported / "encoder.onnx"), so,
                                            providers=["CPUExecutionProvider"])
        self.decoder = ort.InferenceSession(str(exported / "decoder.onnx"), so,
                                            providers=["CPUExecutionProvider"])
        self.meta = json.loads((exported / "metadata.json").read_text())
        self.processor = AutoProcessor.from_pretrained(str(HF_DIR), trust_remote_code=True)
        self.fe = self.processor.feature_extractor
        self.tokenizer = self.processor.tokenizer

    # ------------------------------------------------------------------ audio
    def mel(self, pcm: np.ndarray) -> np.ndarray:
        feat = self.fe([pcm], sampling_rate=16000, return_tensors="np",
                       return_attention_mask=False, padding="longest")["input_features"]
        feat = feat.astype(np.float32)
        if feat.shape[-1] < FRAMES:
            feat = np.pad(feat, ((0, 0), (0, 0), (0, FRAMES - feat.shape[-1])))
        return feat[:, :, :FRAMES]

    def audio_tokens_from_hidden(self, hidden: np.ndarray, n_tokens: int) -> np.ndarray:
        return adaptive_avg_pool(hidden, n_tokens)[None]

    def audio_features(self, pcm: np.ndarray) -> np.ndarray:
        pcm = pcm[:MAX_SAMPLES]
        num_mel = math.ceil(len(pcm) / 160)
        n_tokens = audio_token_count(len(pcm))
        keep_rows = min(encoder_rows(num_mel), FRAMES // 100 * 13)
        hidden = self.encoder.run(None, {"audios": self.mel(pcm),
                                         "num_rows": np.array([keep_rows], np.int64)})[0][0]
        # Block-diagonal attention means the rows of a chunk do not depend on the
        # other chunks, so we only keep the rows the real audio length produces
        # (everything past that is padding).
        return self.audio_tokens_from_hidden(hidden[:keep_rows], n_tokens)

    # ------------------------------------------------------------------ text
    def prompt_ids(self, n_audio: int) -> list[int]:
        text = "Please transcribe this audio."
        suffix = self.tokenizer(text, add_special_tokens=False)["input_ids"]
        return [USER, BOS_AUDIO] + [AUDIO_TOKEN_ID] * n_audio + [EOS_AUDIO] + suffix + [ASSISTANT]

    def decode_ids(self, ids: list[int]) -> str:
        return self.tokenizer.decode(ids, skip_special_tokens=True).strip()

    # ------------------------------------------------------------------ run
    def transcribe(self, wav: str, max_new_tokens: int = 128) -> dict:
        t0 = time.perf_counter()
        pcm = load_wav(wav)
        n_audio = audio_token_count(len(pcm))
        audio = self.audio_features(pcm)
        ids = self.prompt_ids(n_audio)
        tokens = self.greedy(ids, audio, max_new_tokens)
        return {"wav": Path(wav).name, "text": self.decode_ids(tokens),
                "tokens": tokens, "audio_tokens": n_audio, "prompt_len": len(ids),
                "seconds": round(time.perf_counter() - t0, 3)}

    def greedy(self, ids: list[int], audio: np.ndarray,
               max_new_tokens: int = 128) -> list[int]:
        max_total_len = int(self.meta["max_total_len"])
        if len(ids) + 1 > max_total_len:
            raise SystemExit(
                f"prompt ({len(ids)}) does not fit the {max_total_len}-slot cache; "
                "use shorter audio or export with a larger --max-total-len")
        max_new_tokens = min(max_new_tokens, max_total_len - len(ids) - 1)
        n_layers = self.meta["num_layers"]
        n_kv = self.meta["num_kv_heads"]
        head_dim = self.meta["head_dim"]
        caches = []
        for _ in range(n_layers):
            caches.append(np.zeros((1, n_kv, max_total_len, head_dim),
                                   dtype=np.float32))
            caches.append(np.zeros((1, n_kv, max_total_len, head_dim),
                                   dtype=np.float32))

        t_prompt = len(ids)
        attn = np.zeros((1, max_total_len), dtype=np.int64)
        attn[0, :t_prompt] = 1
        feeds = {"input_ids": np.array([ids], dtype=np.int64), "audio_hidden": audio,
                 "attention_mask": attn,
                 "cache_position": np.arange(t_prompt, dtype=np.int64)}
        for i in range(n_layers):
            feeds[f"cache_key_{i}"] = caches[2 * i]
            feeds[f"cache_value_{i}"] = caches[2 * i + 1]
        outs = self.decoder.run(None, feeds)
        logits = outs[0][0, -1, :]
        for i in range(n_layers):
            caches[2 * i][:, :, :t_prompt, :] = outs[1 + 2 * i]
            caches[2 * i + 1][:, :, :t_prompt, :] = outs[2 + 2 * i]

        generated: list[int] = []
        pos = t_prompt
        for _ in range(max_new_tokens):
            nxt = int(np.argmax(logits))
            if nxt in EOS_IDS or nxt == PAD_ID:
                break
            generated.append(nxt)
            attn = np.zeros((1, max_total_len), dtype=np.int64)
            attn[0, : pos + 1] = 1
            feeds = {"input_ids": np.array([[nxt]], dtype=np.int64), "audio_hidden": audio,
                     "attention_mask": attn,
                     "cache_position": np.array([pos], dtype=np.int64)}
            for i in range(n_layers):
                feeds[f"cache_key_{i}"] = caches[2 * i]
                feeds[f"cache_value_{i}"] = caches[2 * i + 1]
            outs = self.decoder.run(None, feeds)
            logits = outs[0][0, -1, :]
            for i in range(n_layers):
                caches[2 * i][:, :, pos : pos + 1, :] = outs[1 + 2 * i]
                caches[2 * i + 1][:, :, pos : pos + 1, :] = outs[2 + 2 * i]
            pos += 1
        return generated


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("wavs", nargs="*")
    ap.add_argument("--out", default=str(ROOT / "reports/onnx-reference.json"))
    args = ap.parse_args()
    wavs = args.wavs
    if not wavs:
        raise SystemExit("please pass at least one wav file")
    asr = Audio8Onnx()
    results = []
    for w in wavs:
        r = asr.transcribe(w)
        print(json.dumps({k: r[k] for k in ("wav", "text", "audio_tokens", "prompt_len",
                                           "seconds")}, ensure_ascii=False), flush=True)
        results.append(r)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
