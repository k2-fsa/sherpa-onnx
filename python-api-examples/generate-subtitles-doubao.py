#!/usr/bin/env python3
#
# Copyright (c)  2026  Xiaomi Corporation

"""
This file demonstrates how to use silero-vad + the Doubao (豆包) bigmodel
recording-file recognition API to generate subtitles.

Supported file formats are those supported by ffmpeg; for instance,
*.mov, *.mp4, *.wav, etc.

It is like ./generate-subtitles.py, which uses a local sherpa-onnx model, but
the ASR here is done by a remote API. The VAD is still silero-vad (or ten-vad);
only the ASR backend is different.

Flow:
  ffmpeg decodes the input file to 16 kHz mono PCM
    -> silero-vad (or ten-vad) cuts out speech segments
    -> each segment is sent to the API as base64 encoded wav data
    -> results are written to <input>.srt and <input>.txt

Please visit
https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/silero_vad.onnx
to download silero_vad.onnx

For instance,

wget https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/silero_vad.onnx

or download ten-vad.onnx, for instance

wget https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/ten-vad.onnx

Please replace --silero-vad-model with --ten-vad-model below to use ten-vad.

You also need an API key of the Doubao bigmodel recording-file recognition
service. Please see https://www.volcengine.com/docs/6561/1354868 for how to
get it. Either export it:

  export DOUBAO_API_KEY=<your_api_key>

or pass it via --api-key.

Finally, please install the python package `requests`:

pip install requests

(1) Process only the first 180 seconds of the input file

./python-api-examples/generate-subtitles-doubao.py  \
  --silero-vad-model=./silero_vad.onnx \
  --max-duration-seconds=180 \
  /path/to/test.mp4

(2) Process the whole input file

./python-api-examples/generate-subtitles-doubao.py  \
  --silero-vad-model=./silero_vad.onnx \
  --max-duration-seconds=0 \
  /path/to/test.mp4

Note: To avoid overwriting existing results, the script refuses to run if
<input>.srt or <input>.txt already exists. Please delete or rename them first.

Note: Speech segments are recognized in parallel; please use --num-workers to
control the number of requests sent to the API at the same time.
"""

import argparse
import base64
import datetime as dt
import io
import os
import shutil
import subprocess
import sys
import time
import uuid
import wave
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path

import numpy as np

try:
    import requests
except ImportError:
    sys.exit("Please install requests first! pip install requests")

import sherpa_onnx

SUBMIT_URL = "https://openspeech.bytedance.com/api/v3/auc/bigmodel/submit"
QUERY_URL = "https://openspeech.bytedance.com/api/v3/auc/bigmodel/query"

# X-Api-Status-Code returned in the response headers
STATUS_OK = "20000000"
STATUS_RUNNING = ("20000001", "20000002")  # processing / queued
STATUS_SILENCE = "20000003"


def get_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--silero-vad-model",
        type=str,
        help="Path to silero_vad.onnx.",
    )

    parser.add_argument(
        "--ten-vad-model",
        type=str,
        help="Path to ten-vad.onnx",
    )

    parser.add_argument(
        "--api-key",
        type=str,
        default="",
        help="""Doubao API key. If not set, the DOUBAO_API_KEY or X_API_KEY
        environment variable is used.""",
    )

    parser.add_argument(
        "--resource-id",
        type=str,
        default="volc.seedasr.auc",
        help="X-Api-Resource-Id of the Doubao ASR service",
    )

    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="Number of parallel requests sent to the Doubao API",
    )

    parser.add_argument(
        "--max-duration-seconds",
        type=float,
        default=180,
        help="""Only process the first N seconds of the input file. Use 0 to
        process the whole file. The default is for debugging only.""",
    )

    parser.add_argument(
        "--enable-itn",
        type=bool,
        default=True,
        help="True to enable inverse text normalization, e.g., 一九九六年 -> 1996年",
    )

    parser.add_argument(
        "--enable-punc",
        type=bool,
        default=True,
        help="True to let the model add punctuation",
    )

    parser.add_argument(
        "--query-interval",
        type=float,
        default=0.5,
        help="Interval in seconds between two query calls",
    )

    parser.add_argument(
        "--query-timeout",
        type=float,
        default=120,
        help="Max seconds to wait for one segment to be transcribed",
    )

    parser.add_argument(
        "--sample-rate",
        type=int,
        default=16000,
        help="Sample rate of the audio sent to the API",
    )

    parser.add_argument(
        "sound_file",
        type=str,
        help="The input sound file to generate subtitles ",
    )

    return parser.parse_args()


def assert_file_exists(filename: str):
    assert Path(filename).exists(), (
        f"{filename} does not exist!\n"
        "Please refer to "
        "https://k2-fsa.github.io/sherpa/onnx/pretrained_models/index.html to download it"
    )


@dataclass
class Segment:
    start: float
    duration: float
    text: str = ""

    @property
    def end(self):
        return self.start + self.duration

    def __str__(self):
        s = f"{timedelta(seconds=self.start)}"[:-3]
        s += " --> "
        s += f"{timedelta(seconds=self.end)}"[:-3]
        s = s.replace(".", ",")
        s += "  "
        s += self.text
        return s

    def to_srt_time(self) -> str:
        s = f"{timedelta(seconds=self.start)}"[:-3]
        s += " --> "
        s += f"{timedelta(seconds=self.end)}"[:-3]
        return s.replace(".", ",")


def samples_to_wav(samples, sample_rate: int) -> bytes:
    """Convert float32 samples in [-1, 1] to a wav file in memory."""
    samples = np.asarray(samples, dtype=np.float32)
    pcm = np.clip(samples * 32767.0, -32768, 32767).astype(np.int16)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as f:
        f.setnchannels(1)
        f.setsampwidth(2)  # int16
        f.setframerate(sample_rate)
        f.writeframes(pcm.tobytes())
    return buf.getvalue()


class DoubaoAsrClient:
    """A thin client for the Doubao bigmodel recording-file recognition API.

    The API is asynchronous: submit a task, then query it until it is done.
    The audio is uploaded as base64 encoded wav data, so no public URL is
    needed.
    """

    def __init__(
        self,
        api_key: str,
        resource_id: str,
        enable_itn: bool = True,
        enable_punc: bool = True,
        query_interval: float = 0.5,
        query_timeout: float = 120,
    ):
        assert api_key, "Empty API key"
        self.api_key = api_key
        self.resource_id = resource_id
        self.enable_itn = enable_itn
        self.enable_punc = enable_punc
        self.query_interval = query_interval
        self.query_timeout = query_timeout

    def _headers(self, request_id: str, log_id: str = "") -> dict:
        headers = {
            "Content-Type": "application/json",
            "X-Api-Key": self.api_key,
            "X-Api-Resource-Id": self.resource_id,
            "X-Api-Request-Id": request_id,
            "X-Api-Sequence": "-1",
        }
        if log_id:
            # The query call must bring back the log id of the submit call
            headers["X-Tt-Logid"] = log_id
        return headers

    def _submit(self, request_id: str, wav_bytes: bytes) -> str:
        """Submit a task and return the X-Tt-Logid of the response."""
        body = {
            "user": {"uid": "sherpa-onnx"},
            "audio": {
                "data": base64.b64encode(wav_bytes).decode("ascii"),
                "format": "wav",
                "codec": "raw",
                "rate": 16000,
                "bits": 16,
                "channel": 1,
            },
            "request": {
                "model_name": "bigmodel",
                "enable_itn": self.enable_itn,
                "enable_punc": self.enable_punc,
                "enable_ddc": False,
                "enable_speaker_info": False,
                "enable_channel_split": False,
                "show_utterances": True,
                "vad_segment": False,
                "sensitive_words_filter": "",
            },
        }

        num_tries = 3
        for i in range(num_tries):
            try:
                response = requests.post(
                    SUBMIT_URL,
                    json=body,
                    headers=self._headers(request_id),
                    timeout=60,
                )
            except requests.RequestException as e:
                if i == num_tries - 1:
                    raise RuntimeError(f"submit failed: {e}") from e
                time.sleep(1 + i)
                continue

            status = response.headers.get("X-Api-Status-Code", "")
            if status == STATUS_OK:
                return response.headers.get("X-Tt-Logid", "")

            message = response.headers.get("X-Api-Message", "")
            if i == num_tries - 1:
                raise RuntimeError(
                    f"submit failed: X-Api-Status-Code={status}, "
                    f"X-Api-Message={message}, body={response.text[:200]}"
                )
            time.sleep(1 + i)

        return ""  # unreachable

    def _query(self, request_id: str, log_id: str) -> dict:
        """Poll the task until it is done and return the result dict."""
        deadline = time.time() + self.query_timeout
        while time.time() < deadline:
            try:
                response = requests.post(
                    QUERY_URL,
                    json={},
                    headers=self._headers(request_id, log_id),
                    timeout=60,
                )
            except requests.RequestException as e:
                raise RuntimeError(f"query failed: {e}") from e

            status = response.headers.get("X-Api-Status-Code", "")
            if status == STATUS_OK:
                return response.json()
            if status == STATUS_SILENCE:
                return {"result": {"text": ""}}
            if status not in STATUS_RUNNING:
                message = response.headers.get("X-Api-Message", "")
                raise RuntimeError(
                    f"query failed: X-Api-Status-Code={status}, "
                    f"X-Api-Message={message}, body={response.text[:200]}"
                )

            time.sleep(self.query_interval)

        raise RuntimeError(
            f"query timed out after {self.query_timeout} seconds: {request_id}"
        )

    def recognize(self, wav_bytes: bytes) -> str:
        """Recognize a single audio clip and return its transcript."""
        request_id = str(uuid.uuid4())
        log_id = self._submit(request_id, wav_bytes)
        response = self._query(request_id, log_id)

        result = response.get("result") or {}
        text = (result.get("text") or "").strip()
        if text:
            return text

        # Fallback: join the texts of the utterances
        utterances = result.get("utterances") or []
        return "".join(u.get("text") or "" for u in utterances).strip()


def create_vad(args) -> tuple[sherpa_onnx.VoiceActivityDetector, int]:
    config = sherpa_onnx.VadModelConfig()
    if args.silero_vad_model:
        config.silero_vad.model = args.silero_vad_model
        config.silero_vad.threshold = 0.1
        config.silero_vad.min_silence_duration = 0.3  # seconds
        config.silero_vad.min_speech_duration = 0.1  # seconds

        # If the current segment is larger than this value, then it increases
        # the threshold to 0.9 internally. After detecting this segment,
        # it resets the threshold to its original value.
        config.silero_vad.max_speech_duration = 15  # seconds
        config.sample_rate = args.sample_rate

        window_size = config.silero_vad.window_size
        print("use silero-vad")
    else:
        config.ten_vad.model = args.ten_vad_model
        config.ten_vad.threshold = 0.2
        config.ten_vad.min_silence_duration = 0.25  # seconds
        config.ten_vad.min_speech_duration = 0.25  # seconds

        # If the current segment is larger than this value, then it increases
        # the threshold to 0.9 internally. After detecting this segment,
        # it resets the threshold to its original value.
        config.ten_vad.max_speech_duration = 5  # seconds
        config.sample_rate = args.sample_rate

        window_size = config.ten_vad.window_size
        print("use ten-vad")

    vad = sherpa_onnx.VoiceActivityDetector(config, buffer_size_in_seconds=100)
    return vad, window_size


def main():
    args = get_args()
    if args.silero_vad_model:
        assert_file_exists(args.silero_vad_model)
    elif args.ten_vad_model:
        assert_file_exists(args.ten_vad_model)
    else:
        raise ValueError("You need to supply one vad model")

    assert args.num_workers > 0, args.num_workers
    assert (
        args.sample_rate == 16000
    ), f"Only 16000 is supported. Given: {args.sample_rate}"

    if not Path(args.sound_file).is_file():
        raise ValueError(f"{args.sound_file} does not exist")

    srt_filename = Path(args.sound_file).with_suffix(".srt")
    txt_filename = Path(args.sound_file).with_suffix(".txt")

    existing = [p for p in (srt_filename, txt_filename) if p.exists()]
    if existing:
        for p in existing:
            print(f"warning: {p} exists!", file=sys.stderr)
        sys.exit("Please delete or rename the file(s) above first!")

    api_key = (
        args.api_key
        or os.environ.get("DOUBAO_API_KEY")
        or os.environ.get("X_API_KEY", "")
    )
    if not api_key:
        raise ValueError(
            "Please set the API key via --api-key or the DOUBAO_API_KEY "
            "environment variable"
        )

    client = DoubaoAsrClient(
        api_key=api_key,
        resource_id=args.resource_id,
        enable_itn=args.enable_itn,
        enable_punc=args.enable_punc,
        query_interval=args.query_interval,
        query_timeout=args.query_timeout,
    )

    max_duration = args.max_duration_seconds
    if max_duration and max_duration > 0:
        print(
            f"Only the first {max_duration} seconds are processed! "
            "Use --max-duration-seconds=0 for the whole file."
        )
    else:
        max_duration = 0
        print("The whole file is processed!")

    ffmpeg_cmd = [
        "ffmpeg",
        "-i",
        args.sound_file,
    ]
    if max_duration > 0:
        ffmpeg_cmd += ["-t", f"{max_duration}"]
    ffmpeg_cmd += [
        "-f",
        "s16le",
        "-acodec",
        "pcm_s16le",
        "-ac",
        "1",
        "-ar",
        str(args.sample_rate),
        "-",
    ]

    process = subprocess.Popen(
        ffmpeg_cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL
    )

    frames_per_read = int(args.sample_rate * 100)  # 100 second

    vad, window_size = create_vad(args)

    buffer = []
    segment_list = []

    print("Started!")
    start_t = dt.datetime.now()
    num_processed_samples = 0

    # Each entry is a (Segment, Future) pair. The future returns the transcript.
    pending = []
    executor = ThreadPoolExecutor(max_workers=args.num_workers)

    def submit_segment(samples: np.ndarray) -> "concurrent.futures.Future[str]":
        wav_bytes = samples_to_wav(samples, args.sample_rate)
        return executor.submit(client.recognize, wav_bytes)

    is_eof = False
    while not is_eof:
        # *2 because int16_t has two bytes
        data = process.stdout.read(frames_per_read * 2)
        if not data:
            vad.flush()
            is_eof = True
        else:
            samples = np.frombuffer(data, dtype=np.int16)
            samples = samples.astype(np.float32) / 32768

            num_processed_samples += samples.shape[0]

            buffer = np.concatenate([buffer, samples])
            while len(buffer) > window_size:
                vad.accept_waveform(buffer[:window_size])
                buffer = buffer[window_size:]

        while not vad.empty():
            segment = Segment(
                start=vad.front.start / args.sample_rate,
                duration=len(vad.front.samples) / args.sample_rate,
            )
            pending.append((segment, submit_segment(vad.front.samples)))
            vad.pop()

    print(f"VAD done! Got {len(pending)} segments. Waiting for the ASR results...")

    for i, (segment, future) in enumerate(pending):
        try:
            segment.text = future.result()
        except Exception as e:
            print(f"Failed to recognize segment {i}: {e}", file=sys.stderr)
            segment.text = ""
        if segment.text:
            segment_list.append(segment)
        print(f"[{i+1}/{len(pending)}] {segment.start:.2f}s {segment.text}")

    executor.shutdown(wait=True)

    end_t = dt.datetime.now()
    elapsed_seconds = (end_t - start_t).total_seconds()
    duration = num_processed_samples / args.sample_rate
    rtf = elapsed_seconds / duration if duration > 0 else 0

    with open(srt_filename, "w", encoding="utf-8") as f:
        for i, seg in enumerate(segment_list):
            print(i + 1, file=f)
            print(seg.to_srt_time(), file=f)
            print(seg.text, file=f)
            print("", file=f)

    print(f"Saved to {srt_filename}")

    with open(txt_filename, "w", encoding="utf-8") as f:
        for seg in segment_list:
            print(seg, file=f)

    print(f"Saved to {txt_filename}")
    print(f"Audio duration:\t{duration:.3f} s")
    print(f"Elapsed:\t{elapsed_seconds:.3f} s")
    print(f"RTF = {elapsed_seconds:.3f}/{duration:.3f} = {rtf:.3f}")
    print("Done!")


if __name__ == "__main__":
    if shutil.which("ffmpeg") is None:
        sys.exit("Please install ffmpeg first!")
    main()
