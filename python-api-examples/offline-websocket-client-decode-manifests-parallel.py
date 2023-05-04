#!/usr/bin/env python3
# Copyright      2022  Xiaomi Corp.        (authors: Fangjun Kuang)
#                2023  Nvidia              (authors: Yuekai Zhang)
# See LICENSE for clarification regarding multiple authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
This script supports to load manifest files in kaldi format and sends it to the server
for decoding, in parallel.

Usage:
# For offline websocket server

python3 client.py \
    --compute-cer \
    --num-tasks $num_task \
    --manifest-dir ./datasets/aishell1_test
"""

import argparse
import asyncio
import json
import math
import os
import time
import types
from pathlib import Path
import wave
import numpy as np
from typing import Tuple
import websockets

from utils import (
    download_and_extract,
    store_transcripts,
    write_error_stats,
    write_triton_stats,
)

DEFAULT_MANIFEST_DIR = "./datasets/aishell1_test"


def get_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--server-addr",
        type=str,
        default="localhost",
        help="Address of the server",
    )

    parser.add_argument(
        "--server-port",
        type=int,
        default=8001,
        help="Grpc port of the triton server, default is 8001",
    )

    parser.add_argument(
        "--manifest-dir",
        type=str,
        default=DEFAULT_MANIFEST_DIR,
        help="Path to the manifest dir which includes wav.scp trans.txt files.",
    )

    parser.add_argument(
        "--audio-path",
        type=str,
        help="Path to a single audio file. It can't be specified at the same time with --manifest-dir",
    )

    parser.add_argument(
        "--num-tasks",
        type=int,
        default=50,
        help="Number of concurrent tasks for sending",
    )

    parser.add_argument(
        "--log-interval",
        type=int,
        default=5,
        help="Controls how frequently we print the log.",
    )

    parser.add_argument(
        "--compute-cer",
        action="store_true",
        default=False,
        help="""True to compute CER, e.g., for Chinese.
        False to compute WER, e.g., for English words.
        """,
    )

    return parser.parse_args()


def load_manifests(dir_path):
    dir_path = Path(dir_path)
    wav_scp_path = dir_path / "wav.scp"
    transcripts_path = dir_path / "trans.txt"

    # Check if the files exist, and raise an error if they don't
    if not wav_scp_path.exists():
        raise ValueError(f"{wav_scp_path} does not exist")
    if not transcripts_path.exists():
        raise ValueError(f"{transcripts_path} does not exist")

    # Load the audio file paths into a dictionary
    with open(wav_scp_path, "r") as f:
        wav_dict = {}
        for line in f:
            line = line.strip()
            if len(line) == 0:
                continue
            parts = line.split()
            if len(parts) != 2:
                raise ValueError(f"Invalid line: {line}")
            wav_dict[parts[0]] = parts[1]

    # Load the transcripts into a dictionary
    with open(transcripts_path, "r") as f:
        trans_dict = {}
        for line in f:
            line = line.strip()
            if len(line) == 0:
                continue
            parts = line.split()
            if len(parts) < 2:
                raise ValueError(f"Invalid line: {line}")
            trans_dict[parts[0]] = " ".join(parts[1:])

    # Combine the two dictionaries into a list of dictionaries
    data = []
    for k, v in wav_dict.items():
        assert k in trans_dict, f"Could not find transcript for {k}"
        data.append(
            {"audio_filepath": str(dir_path / v), "text": trans_dict[k], "id": k}
        )

    return data


def split_data(data, k):
    n = len(data)
    if n < k:
        print(
            f"Warning: the length of the input list ({n}) is less than k ({k}). Setting k to {n}."
        )
        k = n

    quotient = n // k
    remainder = n % k

    result = []
    start = 0
    for i in range(k):
        if i < remainder:
            end = start + quotient + 1
        else:
            end = start + quotient

        result.append(data[start:end])
        start = end

    return result


def read_wave(wave_filename: str) -> Tuple[np.ndarray, int]:
    """
    Args:
      wave_filename:
        Path to a wave file. It should be single channel and each sample should
        be 16-bit. Its sample rate does not need to be 16kHz.
    Returns:
      Return a tuple containing:
       - A 1-D array of dtype np.float32 containing the samples, which are
       normalized to the range [-1, 1].
       - sample rate of the wave file
    """

    with wave.open(wave_filename) as f:
        assert f.getnchannels() == 1, f.getnchannels()
        assert f.getsampwidth() == 2, f.getsampwidth()  # it is in bytes
        num_samples = f.getnframes()
        samples = f.readframes(num_samples)
        samples_int16 = np.frombuffer(samples, dtype=np.int16)
        samples_float32 = samples_int16.astype(np.float32)

        samples_float32 = samples_float32 / 32768

        # duration = samples_float32.shape[0] / f.getframerate()
        # padding_duration = 10
        # padding_length = int(padding_duration * f.getframerate() * ((duration // padding_duration) + 1) - samples_float32.shape[0])
        # samples_pad = np.pad(samples_float32, (0, padding_length), mode='constant')
        # samples_float32 = samples_pad

        return samples_float32, f.getframerate()


async def send_websocket(
    dps: list,
    name: str,
    log_interval: int,
    compute_cer: bool,
    url_addr: str = "localhost:6006",
):
    total_duration = 0.0
    results = []

    async with websockets.connect(
        f"ws://{url_addr}"
    ) as websocket:  # noqa
        for i, dp in enumerate(dps):
            if i % log_interval == 0:
                print(f"{name}: {i}/{len(dps)}")
            samples, sample_rate = read_wave(dp["audio_filepath"])
            duration = len(samples) / sample_rate
            total_duration += duration
            assert isinstance(sample_rate, int)
            assert samples.dtype == np.float32, samples.dtype
            assert samples.ndim == 1, samples.dim
            buf = sample_rate.to_bytes(4, byteorder="little")  # 4 bytes
            buf += (samples.size * 4).to_bytes(4, byteorder="little")
            buf += samples.tobytes()

            await websocket.send(buf)

            decoding_results = await websocket.recv()

            if compute_cer:
                ref = dp["text"].split()
                hyp = decoding_results.split()
                ref = list("".join(ref))
                hyp = list("".join(hyp))
                results.append((dp["id"], ref, hyp))
            else:
                results.append(
                    (
                        dp["id"],
                        dp["text"].split(),
                        decoding_results.split(),
                    )
                )

        # to signal that the client has sent all the data
        await websocket.send("Done")

    return total_duration, results

async def main():
    args = get_args()
    if args.audio_path:
        args.num_tasks = 1
        args.log_interval = 1
        dps_list = [
            [{
                "audio_filepath": args.audio_path,
                "text": "foo",
                "id": 0,
            }]
        ]
    else:
        if not any(Path(args.manifest_dir).rglob("*.wav")):
            if args.manifest_dir == DEFAULT_MANIFEST_DIR:
                download_and_extract(args.manifest_dir)
            raise ValueError(
                f"manifest_dir {args.manifest_dir} should contain wav files"
            )
        dps = load_manifests(args.manifest_dir)
        dps_list = split_data(dps, args.num_tasks)
        args.num_tasks = min(args.num_tasks, len(dps_list))

    url = f"{args.server_addr}:{args.server_port}"

    tasks = []
    start_time = time.time()
    for i in range(args.num_tasks):
        task = asyncio.create_task(
            send_websocket(
                dps=dps_list[i],
                name=f"task-{i}",
                log_interval=args.log_interval,
                compute_cer=args.compute_cer,
            )
        )
        tasks.append(task)

    ans_list = await asyncio.gather(*tasks)

    end_time = time.time()
    elapsed = end_time - start_time

    results = []
    total_duration = 0.0
    latency_data = []
    for ans in ans_list:
        total_duration += ans[0]
        results += ans[1]

    rtf = elapsed / total_duration

    s = f"RTF: {rtf:.4f}\n"
    s += f"total_duration: {total_duration:.3f} seconds\n"
    s += f"({total_duration/3600:.2f} hours)\n"
    s += f"processing time: {elapsed:.3f} seconds " f"({elapsed/3600:.2f} hours)\n"

    print(s)

    with open("rtf.txt", "w") as f:
        f.write(s)

    name = Path(args.manifest_dir).stem.split(".")[0]
    results = sorted(results)
    store_transcripts(filename=f"recogs-{name}.txt", texts=results)

    with open(f"errs-{name}.txt", "w") as f:
        write_error_stats(f, "test-set", results, enable_log=True)

    with open(f"errs-{name}.txt", "r") as f:
        print(f.readline())  # WER
        print(f.readline())  # Detailed errors

if __name__ == "__main__":
    asyncio.run(main())
