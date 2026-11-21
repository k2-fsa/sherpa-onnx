#!/usr/bin/env python3
"""Benchmark a local sherpa-onnx build against public 16 kHz PCM16 WAV files.

Run each configuration in a fresh process, using an independent warmup run
and three measured runs with distinct --output paths. Each JSON report gives
model/runtime metadata, fixture IDs and SHA-256 hashes, corpus S/D/I/N and WER
(as a fraction), recognition wall time and RTF, nearest-rank decode p50/p95
(in ms), model load time (in seconds), and process peak RSS (in bytes).
Recognition time includes stream creation, audio ingestion, endpoint handling,
and EOF drain. It excludes disk reads, text scoring, and input hashing.
"""

import argparse
import hashlib
import json
import math
import platform
import re
import sys
import time
import wave
from array import array
from pathlib import Path
from typing import List, Sequence, Tuple

SAMPLE_RATE = 16000
PACKET_MS = 20
PACKET_SAMPLES = SAMPLE_RATE * PACKET_MS // 1000
MAX_WAV_FRAMES = SAMPLE_RATE * 300
MAX_FIXTURES = 1000
MAX_LINE_BYTES = 16384
MAX_REFERENCE_WORDS = 1024
MAX_HYPOTHESIS_WORDS = 4096
WORD_PATTERN = re.compile(r"[^\W_]+(?:'[^\W_]+)*")
FIXTURE_ID_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}")
WAV_PATH_PATTERN = re.compile(r"[A-Za-z0-9_./ -]{1,512}")
Fixture = Tuple[str, Path, List[str]]


def words(value: str) -> List[str]:
    """Lowercase and split on punctuation/whitespace, keeping internal apostrophes."""
    return WORD_PATTERN.findall(value.lower().replace("’", "'"))


def edit_counts(reference: List[str], hypothesis: List[str]) -> Tuple[int, int, int]:
    """Return minimum-distance substitutions, deletions, and insertions."""
    # Each cell holds (distance, S, D, I). Ties favor a diagonal move, then
    # a deletion, then an insertion, in that order.
    previous = [(i, 0, 0, i) for i in range(len(hypothesis) + 1)]
    for i, ref_word in enumerate(reference, 1):
        current = [(i, 0, i, 0)]
        for j, hyp_word in enumerate(hypothesis, 1):
            distance, subs, deletes, inserts = previous[j - 1]
            if ref_word == hyp_word:
                current.append(previous[j - 1])
                continue
            substitution = (distance + 1, subs + 1, deletes, inserts)
            distance, subs, deletes, inserts = previous[j]
            deletion = (distance + 1, subs, deletes + 1, inserts)
            distance, subs, deletes, inserts = current[-1]
            insertion = (distance + 1, subs, deletes, inserts + 1)
            current.append(min((substitution, deletion, insertion), key=lambda x: x[0]))
        previous = current
    return previous[-1][1:]


def input_file(path: Path, suffix: str, max_bytes: int) -> Path:
    if ".." in path.parts or path.is_symlink() or path.suffix.lower() != suffix:
        raise ValueError("Invalid input file path or extension")
    resolved = path.resolve(strict=True)
    if not resolved.is_file() or not 0 < resolved.stat().st_size <= max_bytes:
        raise ValueError("Input must be a regular file within the size limit")
    return resolved


def manifest_record(pairs: List[Tuple[str, str]]) -> dict:
    if len(pairs) != 3 or {key for key, _ in pairs} != {"id", "wav", "text"}:
        raise ValueError("Expected exactly the manifest fields id, wav, and text")
    record = dict(pairs)
    if any(not isinstance(value, str) for value in record.values()):
        raise ValueError("Manifest fields must be strings")
    return record


def read_manifest(path: Path) -> List[Fixture]:
    fixtures = []
    fixture_ids = set()
    base = path.parent.resolve(strict=True)
    with path.open("rb") as source:
        for row in range(1, MAX_FIXTURES + 2):
            line = source.readline(MAX_LINE_BYTES + 1)
            if not line:
                break
            try:
                if row > MAX_FIXTURES or len(line) > MAX_LINE_BYTES:
                    raise ValueError
                record = json.loads(
                    line.decode("utf-8"), object_pairs_hook=manifest_record
                )
                if not isinstance(record, dict):
                    raise ValueError
                fixture_id = record["id"]
                if (
                    not FIXTURE_ID_PATTERN.fullmatch(fixture_id)
                    or fixture_id in fixture_ids
                ):
                    raise ValueError
                wav_name = record["wav"]
                relative_path = Path(wav_name)
                if (
                    not WAV_PATH_PATTERN.fullmatch(wav_name)
                    or relative_path.is_absolute()
                    or ".." in relative_path.parts
                ):
                    raise ValueError
                wav_path = base
                for part in relative_path.parts:
                    wav_path /= part
                    if wav_path.is_symlink():
                        raise ValueError
                wav_path = input_file(wav_path, ".wav", MAX_WAV_FRAMES * 2 + 65536)
                wav_path.relative_to(base)
                reference = words(record["text"])
                if not 0 < len(reference) <= MAX_REFERENCE_WORDS:
                    raise ValueError
                fixtures.append((fixture_id, wav_path, reference))
                fixture_ids.add(fixture_id)
            except (OSError, RecursionError, ValueError):
                raise ValueError(f"Invalid manifest row {row}") from None
    if not fixtures:
        raise ValueError("Manifest must contain at least one fixture")
    return fixtures


def read_wav(path: Path) -> array:
    try:
        with wave.open(str(path), "rb") as source:
            frames = source.getnframes()
            if (
                source.getnchannels() != 1
                or source.getsampwidth() != 2
                or source.getframerate() != SAMPLE_RATE
                or source.getcomptype() != "NONE"
                or not 0 < frames <= MAX_WAV_FRAMES
            ):
                raise ValueError
            data = source.readframes(frames)
            if len(data) != frames * 2:
                raise ValueError
    except (EOFError, OSError, ValueError, wave.Error):
        raise ValueError(
            "Expected a complete mono 16 kHz PCM16 WAV of up to 300 seconds"
        ) from None
    pcm = array("h")
    pcm.frombytes(data)
    if sys.byteorder == "big":
        pcm.byteswap()
    return array("f", (sample / 32768.0 for sample in pcm))


def transcribe(
    recognizer, samples: Sequence[float], endpoint: bool, decode_seconds: List[float]
) -> str:
    stream = recognizer.create_stream()
    segments = []

    def drain():
        while recognizer.is_ready(stream):
            start = time.perf_counter()
            recognizer.decode_stream(stream)
            decode_seconds.append(time.perf_counter() - start)
            if endpoint and recognizer.is_endpoint(stream):
                segments.append(recognizer.get_result(stream))
                recognizer.reset(stream)

    for offset in range(0, len(samples), PACKET_SAMPLES):
        stream.accept_waveform(SAMPLE_RATE, samples[offset : offset + PACKET_SAMPLES])
        drain()
    stream.input_finished()
    drain()
    segments.append(recognizer.get_result(stream))
    return " ".join(segment for segment in segments if segment)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def benchmark(args: argparse.Namespace) -> dict:
    import resource

    import sherpa_onnx

    models = {
        "encoder": input_file(args.encoder, ".onnx", 8 * 1024**3),
        "decoder": input_file(args.decoder, ".onnx", 8 * 1024**3),
        "joiner": input_file(args.joiner, ".onnx", 8 * 1024**3),
        "tokens": input_file(args.tokens, ".txt", 16 * 1024**2),
    }
    manifest = input_file(args.manifest, ".jsonl", MAX_FIXTURES * MAX_LINE_BYTES)
    fixtures = read_manifest(manifest)

    start = time.perf_counter()
    recognizer = sherpa_onnx.OnlineRecognizer.from_transducer(
        **{name: str(path) for name, path in models.items()},
        provider="cpu",
        sample_rate=SAMPLE_RATE,
        num_threads=args.num_threads,
        decoding_method=args.method,
        max_active_paths=args.beam,
        enable_endpoint_detection=args.endpoint == "on",
    )
    model_load_seconds = time.perf_counter() - start

    decode_seconds = []
    recognition_seconds = 0.0
    total_samples = 0
    substitutions = deletions = insertions = reference_words = 0
    for _, path, reference in fixtures:
        samples = read_wav(path)
        start = time.perf_counter()
        result = transcribe(recognizer, samples, args.endpoint == "on", decode_seconds)
        recognition_seconds += time.perf_counter() - start
        hypothesis = words(result)
        if len(hypothesis) > MAX_HYPOTHESIS_WORDS:
            raise ValueError("Recognition exceeded the word limit")
        s, d, i = edit_counts(reference, hypothesis)
        substitutions += s
        deletions += d
        insertions += i
        reference_words += len(reference)
        total_samples += len(samples)

    peak_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if sys.platform != "darwin":
        peak_rss *= 1024
    audio_seconds = total_samples / SAMPLE_RATE
    decode_seconds.sort()
    return {
        "config": {
            "method": args.method,
            "beam": args.beam,
            "num_threads": args.num_threads,
            "endpoint": args.endpoint,
            "provider": "cpu",
            "sample_rate": SAMPLE_RATE,
            "packet_ms": PACKET_MS,
        },
        "versions": {
            "sherpa_onnx": sherpa_onnx.version,
            "git_sha1": sherpa_onnx.git_sha1,
            "onnxruntime": sherpa_onnx.onnxruntime_version,
            "python": platform.python_version(),
            "system": platform.system(),
            "machine": platform.machine(),
        },
        "model_sha256": {name: sha256(path) for name, path in models.items()},
        "manifest_sha256": sha256(manifest),
        "fixtures": [
            {"id": fixture_id, "wav_sha256": sha256(path)}
            for fixture_id, path, _ in fixtures
        ],
        "wer": {
            "S": substitutions,
            "D": deletions,
            "I": insertions,
            "N": reference_words,
            "rate": (substitutions + deletions + insertions) / reference_words,
        },
        "audio_seconds": audio_seconds,
        "recognition_seconds": recognition_seconds,
        "rtf": recognition_seconds / audio_seconds,
        "decode_calls": len(decode_seconds),
        "decode_p50_ms": percentile_ms(decode_seconds, 0.50),
        "decode_p95_ms": percentile_ms(decode_seconds, 0.95),
        "model_load_seconds": model_load_seconds,
        "peak_rss_bytes": peak_rss,
    }


def percentile_ms(sorted_seconds: List[float], percentile: float):
    if not sorted_seconds:
        return None
    return sorted_seconds[math.ceil(percentile * len(sorted_seconds)) - 1] * 1000


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__.split("\n\n", 1)[0],
        epilog=(
            "JSON output includes fixture and model hashes, WER counts and rate, "
            "RTF, per-decode p50/p95 in ms, load time, and process peak RSS in bytes."
        ),
    )
    for name in ("encoder", "decoder", "joiner", "tokens", "manifest"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument(
        "--method", choices=("greedy_search", "modified_beam_search"), required=True
    )
    parser.add_argument("--beam", type=int, default=4, help="active paths (1–32)")
    parser.add_argument(
        "--num-threads", type=int, default=2, help="CPU threads (1–64)"
    )
    parser.add_argument("--endpoint", choices=("on", "off"), required=True)
    parser.add_argument(
        "--output", type=Path, required=True, help="new JSON output file"
    )
    args = parser.parse_args()
    if not 1 <= args.beam <= 32 or not 1 <= args.num_threads <= 64:
        parser.error("Expected beam 1–32 and num-threads 1–64")
    if sys.platform not in ("darwin", "linux"):
        parser.error("Peak RSS reporting requires macOS or Linux")
    try:
        if ".." in args.output.parts or args.output.suffix.lower() != ".json":
            raise ValueError(
                "Output must have a .json extension and no parent traversal"
            )
        output = args.output.parent.resolve(strict=True) / args.output.name
        if output.exists() or output.is_symlink():
            raise ValueError("Output must be a new file")
        report = benchmark(args)
        with output.open("x", encoding="utf-8") as destination:
            json.dump(report, destination, indent=2, allow_nan=False)
            destination.write("\n")
    except ImportError:
        parser.error("Could not import the local sherpa-onnx Python build")
    except ValueError as exc:
        parser.error(str(exc))
    except OSError:
        parser.error("Could not read an input file or create the report")


if __name__ == "__main__":
    main()
