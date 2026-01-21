#!/usr/bin/env python3
# /// script
# dependencies = ["numpy", "jiwer"]
# ///
from __future__ import annotations

"""
Run timestamp accuracy benchmark against LibriSpeech ground truth.

Compares sherpa-onnx Whisper word timestamps against MFA alignments.

Usage:
    PYTHONPATH=build/lib:sherpa-onnx/python python scripts/benchmark/run_timestamp_benchmark.py \
        --encoder ./whisper-tiny-attention/tiny-encoder.onnx \
        --decoder ./whisper-tiny-attention/tiny-decoder.onnx \
        --tokens ./whisper-tiny-attention/tiny-tokens.txt

    # Parallel processing with 4 workers:
    PYTHONPATH=build/lib:sherpa-onnx/python python scripts/benchmark/run_timestamp_benchmark.py \
        --encoder ./whisper-tiny-attention/tiny-encoder.onnx \
        --decoder ./whisper-tiny-attention/tiny-decoder.onnx \
        --tokens ./whisper-tiny-attention/tiny-tokens.txt \
        --num-workers 4

Outputs:
    benchmark_results/details_YYYYMMDD_HHMMSS.csv - Per-word timing errors
    benchmark_results/summary_YYYYMMDD_HHMMSS.csv - Aggregate statistics
"""

import argparse
import csv
import json
import multiprocessing
import os
import re
import sys
import time
import wave
from dataclasses import dataclass
from datetime import datetime
from difflib import SequenceMatcher
from pathlib import Path

import numpy as np

try:
    import sherpa_onnx
except ImportError:
    print("ERROR: sherpa_onnx not found. Please install it using one of the methods at:")
    print("https://k2-fsa.github.io/sherpa/onnx/python/install.html")
    sys.exit(1)

try:
    import jiwer
    from jiwer import wer as compute_wer
except ImportError:
    print("ERROR: jiwer not found. Install with: pip install jiwer")
    sys.exit(1)

# Text normalization for WER calculation
wer_transforms = jiwer.Compose([
    jiwer.ToLowerCase(),
    jiwer.RemovePunctuation(),
    jiwer.ExpandCommonEnglishContractions(),
    jiwer.SubstituteWords({
        "mr": "mister",
        "mrs": "missus",
        "dr": "doctor",
        "st": "saint",
    }),
    jiwer.RemoveMultipleSpaces(),
    jiwer.Strip(),
    jiwer.ReduceToListOfListOfWords(),
])


@dataclass
class WordTiming:
    """A word with its timing information."""
    word: str
    start: float
    end: float


@dataclass
class AlignedWord:
    """A pair of ground truth and predicted words that have been aligned."""
    word: str
    gt_start: float
    gt_end: float
    pred_start: float | None
    pred_end: float | None
    matched: bool


def normalize_word(word: str) -> str:
    """Normalize word for comparison."""
    # Remove punctuation, lowercase
    return re.sub(r'[^\w]', '', word).strip().lower()


def read_wave(wave_filename: str) -> tuple[np.ndarray, int]:
    """Read a wave file and return samples as float32 array."""
    with wave.open(wave_filename) as f:
        assert f.getnchannels() == 1, f"Expected mono, got {f.getnchannels()} channels"
        assert f.getsampwidth() == 2, f"Expected 16-bit, got {f.getsampwidth() * 8}-bit"
        num_samples = f.getnframes()
        samples = f.readframes(num_samples)
        samples_int16 = np.frombuffer(samples, dtype=np.int16)
        samples_float32 = samples_int16.astype(np.float32) / 32768
        return samples_float32, f.getframerate()


def tokens_to_words(
    tokens: list[str],
    timestamps: list[float],
    durations: list[float]
) -> list[WordTiming]:
    """
    Convert token-level timestamps to word-level timestamps.

    Follows OpenAI Whisper's split_tokens_on_spaces logic:
    - Tokens starting with space begin a new word
    - Punctuation-only tokens begin a new word
    - Otherwise append to previous word
    """
    import string

    if not tokens:
        return []

    words = []
    current_word = ""
    current_start = None
    current_end = None

    for token, ts, dur in zip(tokens, timestamps, durations):
        token_end = ts + dur
        token_stripped = token.strip()

        # Determine if this token starts a new word
        with_space = token.startswith(" ")
        is_punctuation = token_stripped in string.punctuation
        is_first = len(words) == 0 and current_word == ""

        if with_space or is_punctuation or is_first:
            # Save previous word if exists
            if current_word.strip():
                words.append(WordTiming(
                    word=current_word.strip(),
                    start=current_start,
                    end=current_end
                ))
            # Start new word
            current_word = token
            current_start = ts
            current_end = token_end
        else:
            # Append to current word
            current_word += token
            current_end = token_end

    # Don't forget the last word
    if current_word.strip():
        words.append(WordTiming(
            word=current_word.strip(),
            start=current_start,
            end=current_end
        ))

    return words


def get_sherpa_word_timestamps(
    recognizer: sherpa_onnx.OfflineRecognizer,
    audio_path: Path
) -> list[WordTiming]:
    """Run sherpa-onnx recognition and return word timestamps."""
    samples, sample_rate = read_wave(str(audio_path))

    stream = recognizer.create_stream()
    stream.accept_waveform(sample_rate, samples)
    recognizer.decode_stream(stream)
    result = stream.result

    # Convert token timestamps to word timestamps
    return tokens_to_words(result.tokens, result.timestamps, result.durations)


# Global recognizer for worker processes
_worker_recognizer = None


def _init_worker(encoder: str, decoder: str, tokens: str, language: str):
    """Initialize recognizer in worker process."""
    global _worker_recognizer
    _worker_recognizer = sherpa_onnx.OfflineRecognizer.from_whisper(
        encoder=encoder,
        decoder=decoder,
        tokens=tokens,
        language=language,
        enable_token_timestamps=True,
    )


def _process_utterance(args: tuple) -> dict:
    """Process a single utterance in a worker process."""
    item, data_dir = args
    utterance_id = item["utterance_id"]
    audio_path = Path(data_dir) / item["audio_path"]

    # Parse ground truth
    gt_words = [
        WordTiming(word=wt["word"], start=wt["start"], end=wt["end"])
        for wt in item["word_times"]
    ]
    gt_transcript = " ".join(w.word for w in gt_words)

    # Get predictions
    pred_words = get_sherpa_word_timestamps(_worker_recognizer, audio_path)
    pred_transcript = " ".join(w.word for w in pred_words)

    # Align words
    aligned = align_words(gt_words, pred_words)

    # Calculate per-utterance stats
    matched = [a for a in aligned if a.matched]

    if matched:
        start_errors = [abs(a.pred_start - a.gt_start) * 1000 for a in matched]
        end_errors = [abs(a.pred_end - a.gt_end) * 1000 for a in matched]

        stats = {
            "utterance_id": utterance_id,
            "num_gt_words": len(gt_words),
            "num_pred_words": len(pred_words),
            "num_matched": len(matched),
            "match_rate": len(matched) / len(gt_words) if gt_words else 0,
            "wer": jiwer.wer(
                gt_transcript,
                pred_transcript,
                reference_transform=wer_transforms,
                hypothesis_transform=wer_transforms,
            ),
            "mean_start_error_ms": np.mean(start_errors),
            "median_start_error_ms": np.median(start_errors),
            "max_start_error_ms": np.max(start_errors),
            "mean_end_error_ms": np.mean(end_errors),
            "median_end_error_ms": np.median(end_errors),
            "max_end_error_ms": np.max(end_errors),
            "pct_within_20ms": sum(1 for e in start_errors if e <= 20) / len(start_errors) * 100,
            "pct_within_50ms": sum(1 for e in start_errors if e <= 50) / len(start_errors) * 100,
        }
    else:
        stats = {
            "utterance_id": utterance_id,
            "num_gt_words": len(gt_words),
            "num_pred_words": len(pred_words),
            "num_matched": 0,
            "match_rate": 0,
            "wer": jiwer.wer(
                gt_transcript,
                pred_transcript,
                reference_transform=wer_transforms,
                hypothesis_transform=wer_transforms,
            ) if gt_transcript else 1.0,
            "mean_start_error_ms": None,
            "median_start_error_ms": None,
            "max_start_error_ms": None,
            "mean_end_error_ms": None,
            "median_end_error_ms": None,
            "max_end_error_ms": None,
            "pct_within_20ms": None,
            "pct_within_50ms": None,
        }

    # Build aligned words for detailed output
    aligned_words = []
    for j, a in enumerate(aligned):
        aligned_words.append({
            "utterance_id": utterance_id,
            "word_index": j,
            "word": a.word,
            "gt_start": a.gt_start,
            "gt_end": a.gt_end,
            "pred_start": a.pred_start if a.pred_start is not None else "",
            "pred_end": a.pred_end if a.pred_end is not None else "",
            "matched": a.matched,
            "start_error_ms": abs(a.pred_start - a.gt_start) * 1000 if a.matched else "",
            "end_error_ms": abs(a.pred_end - a.gt_end) * 1000 if a.matched else "",
        })

    return {"stats": stats, "aligned": aligned_words}


def align_words(
    gt_words: list[WordTiming],
    pred_words: list[WordTiming]
) -> list[AlignedWord]:
    """
    Align ground truth and predicted words using sequence matching.

    Returns list of AlignedWord with timing comparisons for matched words.
    """
    # Normalize words for matching
    gt_normalized = [normalize_word(w.word) for w in gt_words]
    pred_normalized = [normalize_word(w.word) for w in pred_words]

    # Use SequenceMatcher to find matching blocks
    matcher = SequenceMatcher(None, gt_normalized, pred_normalized)

    aligned = []
    matched_pred_indices = set()

    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == 'equal':
            # Words match
            for gt_idx, pred_idx in zip(range(i1, i2), range(j1, j2)):
                gt_word = gt_words[gt_idx]
                pred_word = pred_words[pred_idx]
                aligned.append(AlignedWord(
                    word=gt_word.word,
                    gt_start=gt_word.start,
                    gt_end=gt_word.end,
                    pred_start=pred_word.start,
                    pred_end=pred_word.end,
                    matched=True
                ))
                matched_pred_indices.add(pred_idx)
        elif tag in ('replace', 'delete'):
            # Ground truth words not matched
            for gt_idx in range(i1, i2):
                gt_word = gt_words[gt_idx]
                aligned.append(AlignedWord(
                    word=gt_word.word,
                    gt_start=gt_word.start,
                    gt_end=gt_word.end,
                    pred_start=None,
                    pred_end=None,
                    matched=False
                ))

    return aligned


def run_benchmark(
    manifest: list[dict],
    data_dir: Path,
    output_dir: Path,
    encoder: str,
    decoder: str,
    tokens: str,
    language: str,
    num_workers: int = 1
):
    """Run benchmark on all utterances in manifest."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    details_path = output_dir / f"details_{timestamp}.csv"
    summary_path = output_dir / f"summary_{timestamp}.csv"

    output_dir.mkdir(parents=True, exist_ok=True)

    # Collect all results
    all_aligned = []
    utterance_stats = []

    total = len(manifest)
    start_time = time.time()

    if num_workers > 1:
        # Parallel processing
        print(f"\nProcessing {total} utterances with {num_workers} workers...")

        # Prepare arguments for workers
        work_items = [(item, str(data_dir)) for item in manifest]

        with multiprocessing.Pool(
            processes=num_workers,
            initializer=_init_worker,
            initargs=(encoder, decoder, tokens, language)
        ) as pool:
            completed = 0
            for result in pool.imap(_process_utterance, work_items):
                utterance_stats.append(result["stats"])
                all_aligned.extend(result["aligned"])

                completed += 1
                elapsed = time.time() - start_time
                avg_per_item = elapsed / completed
                remaining = total - completed
                eta_seconds = avg_per_item * remaining

                if eta_seconds >= 3600:
                    eta_str = f"{eta_seconds / 3600:.1f}h"
                elif eta_seconds >= 60:
                    eta_str = f"{eta_seconds / 60:.1f}m"
                else:
                    eta_str = f"{eta_seconds:.0f}s"

                print(f"  [{completed}/{total}] {result['stats']['utterance_id']} - ETA: {eta_str}", flush=True)
    else:
        # Sequential processing (original behavior)
        print(f"\nProcessing {total} utterances...")

        # Initialize recognizer for sequential mode
        recognizer = sherpa_onnx.OfflineRecognizer.from_whisper(
            encoder=encoder,
            decoder=decoder,
            tokens=tokens,
            language=language,
            enable_token_timestamps=True,
        )

        for i, item in enumerate(manifest):
            iter_start = time.time()
            utterance_id = item["utterance_id"]
            audio_path = data_dir / item["audio_path"]

            # Parse ground truth
            gt_words = [
                WordTiming(word=wt["word"], start=wt["start"], end=wt["end"])
                for wt in item["word_times"]
            ]
            gt_transcript = " ".join(w.word for w in gt_words)

            # Get predictions
            pred_words = get_sherpa_word_timestamps(recognizer, audio_path)
            pred_transcript = " ".join(w.word for w in pred_words)

            # Align words
            aligned = align_words(gt_words, pred_words)

            # Calculate per-utterance stats
            matched = [a for a in aligned if a.matched]

            if matched:
                start_errors = [abs(a.pred_start - a.gt_start) * 1000 for a in matched]
                end_errors = [abs(a.pred_end - a.gt_end) * 1000 for a in matched]

                stats = {
                    "utterance_id": utterance_id,
                    "num_gt_words": len(gt_words),
                    "num_pred_words": len(pred_words),
                    "num_matched": len(matched),
                    "match_rate": len(matched) / len(gt_words) if gt_words else 0,
                    "wer": jiwer.wer(
                        gt_transcript,
                        pred_transcript,
                        reference_transform=wer_transforms,
                        hypothesis_transform=wer_transforms,
                    ),
                    "mean_start_error_ms": np.mean(start_errors),
                    "median_start_error_ms": np.median(start_errors),
                    "max_start_error_ms": np.max(start_errors),
                    "mean_end_error_ms": np.mean(end_errors),
                    "median_end_error_ms": np.median(end_errors),
                    "max_end_error_ms": np.max(end_errors),
                    "pct_within_20ms": sum(1 for e in start_errors if e <= 20) / len(start_errors) * 100,
                    "pct_within_50ms": sum(1 for e in start_errors if e <= 50) / len(start_errors) * 100,
                }
            else:
                stats = {
                    "utterance_id": utterance_id,
                    "num_gt_words": len(gt_words),
                    "num_pred_words": len(pred_words),
                    "num_matched": 0,
                    "match_rate": 0,
                    "wer": jiwer.wer(
                        gt_transcript,
                        pred_transcript,
                        reference_transform=wer_transforms,
                        hypothesis_transform=wer_transforms,
                    ) if gt_transcript else 1.0,
                    "mean_start_error_ms": None,
                    "median_start_error_ms": None,
                    "max_start_error_ms": None,
                    "mean_end_error_ms": None,
                    "median_end_error_ms": None,
                    "max_end_error_ms": None,
                    "pct_within_20ms": None,
                    "pct_within_50ms": None,
                }

            utterance_stats.append(stats)

            # Store aligned words for detailed output
            for j, a in enumerate(aligned):
                all_aligned.append({
                    "utterance_id": utterance_id,
                    "word_index": j,
                    "word": a.word,
                    "gt_start": a.gt_start,
                    "gt_end": a.gt_end,
                    "pred_start": a.pred_start if a.pred_start is not None else "",
                    "pred_end": a.pred_end if a.pred_end is not None else "",
                    "matched": a.matched,
                    "start_error_ms": abs(a.pred_start - a.gt_start) * 1000 if a.matched else "",
                    "end_error_ms": abs(a.pred_end - a.gt_end) * 1000 if a.matched else "",
                })

            # Progress with ETA
            completed = i + 1
            elapsed = time.time() - start_time
            avg_per_item = elapsed / completed
            remaining = total - completed
            eta_seconds = avg_per_item * remaining

            if eta_seconds >= 3600:
                eta_str = f"{eta_seconds / 3600:.1f}h"
            elif eta_seconds >= 60:
                eta_str = f"{eta_seconds / 60:.1f}m"
            else:
                eta_str = f"{eta_seconds:.0f}s"

            iter_time = time.time() - iter_start
            print(f"  [{completed}/{total}] {utterance_id} ({iter_time:.1f}s) - ETA: {eta_str}", flush=True)

    # Sort results by utterance_id to ensure consistent output
    utterance_stats.sort(key=lambda x: x["utterance_id"])
    all_aligned.sort(key=lambda x: (x["utterance_id"], x["word_index"]))

    # Write detailed results
    print(f"\nWriting detailed results to {details_path}...")
    with open(details_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "utterance_id", "word_index", "word", "gt_start", "gt_end",
            "pred_start", "pred_end", "matched", "start_error_ms", "end_error_ms"
        ])
        writer.writeheader()
        writer.writerows(all_aligned)

    # Write summary results
    print(f"Writing summary to {summary_path}...")
    with open(summary_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "utterance_id", "num_gt_words", "num_pred_words", "num_matched",
            "match_rate", "wer", "mean_start_error_ms", "median_start_error_ms",
            "max_start_error_ms", "mean_end_error_ms", "median_end_error_ms",
            "max_end_error_ms", "pct_within_20ms", "pct_within_50ms"
        ])
        writer.writeheader()
        writer.writerows(utterance_stats)

    # Print aggregate stats
    matched_stats = [s for s in utterance_stats if s["num_matched"] > 0]
    if matched_stats:
        print("\n" + "=" * 60)
        print("AGGREGATE RESULTS")
        print("=" * 60)
        print(f"Total utterances: {len(manifest)}")
        print(f"Total ground truth words: {sum(s['num_gt_words'] for s in utterance_stats)}")
        print(f"Total matched words: {sum(s['num_matched'] for s in utterance_stats)}")

        all_start_errors = [
            float(r["start_error_ms"]) for r in all_aligned
            if r["matched"] and r["start_error_ms"] != ""
        ]
        all_end_errors = [
            float(r["end_error_ms"]) for r in all_aligned
            if r["matched"] and r["end_error_ms"] != ""
        ]

        if all_start_errors:
            print(f"\nStart Time Errors:")
            print(f"  Mean: {np.mean(all_start_errors):.1f} ms")
            print(f"  Median: {np.median(all_start_errors):.1f} ms")
            print(f"  Max: {np.max(all_start_errors):.1f} ms")
            print(f"  Std: {np.std(all_start_errors):.1f} ms")

            print(f"\nEnd Time Errors:")
            print(f"  Mean: {np.mean(all_end_errors):.1f} ms")
            print(f"  Median: {np.median(all_end_errors):.1f} ms")
            print(f"  Max: {np.max(all_end_errors):.1f} ms")
            print(f"  Std: {np.std(all_end_errors):.1f} ms")

            print(f"\nAccuracy Thresholds (start time):")
            print(f"  Within 20ms: {sum(1 for e in all_start_errors if e <= 20) / len(all_start_errors) * 100:.1f}%")
            print(f"  Within 50ms: {sum(1 for e in all_start_errors if e <= 50) / len(all_start_errors) * 100:.1f}%")
            print(f"  Within 100ms: {sum(1 for e in all_start_errors if e <= 100) / len(all_start_errors) * 100:.1f}%")

        avg_wer = np.mean([s["wer"] for s in utterance_stats])
        print(f"\nWord Error Rate (WER): {avg_wer * 100:.1f}%")

    return details_path, summary_path


def main():
    parser = argparse.ArgumentParser(description="Run timestamp accuracy benchmark")
    parser.add_argument("--encoder", required=True, help="Path to encoder.onnx")
    parser.add_argument("--decoder", required=True, help="Path to decoder.onnx")
    parser.add_argument("--tokens", required=True, help="Path to tokens.txt")
    parser.add_argument(
        "--data-dir",
        default="benchmark_data",
        help="Directory with manifest.json and audio (default: benchmark_data)"
    )
    parser.add_argument(
        "--output-dir",
        default="benchmark_results",
        help="Output directory for CSV files (default: benchmark_results)"
    )
    parser.add_argument(
        "--language",
        default="en",
        help="Language code (default: en)"
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=1,
        help="Number of parallel workers (default: 1, sequential). "
             "Use higher values to speed up benchmarks on multi-core machines. "
             "Each worker loads its own model copy, so memory usage scales linearly."
    )
    args = parser.parse_args()

    # Resolve paths
    script_dir = Path(__file__).parent.resolve()
    repo_root = script_dir.parent.parent
    data_dir = repo_root / args.data_dir
    output_dir = repo_root / args.output_dir
    manifest_path = data_dir / "manifest.json"

    # Load manifest
    print(f"Loading manifest from {manifest_path}...")
    with open(manifest_path) as f:
        manifest = json.load(f)
    print(f"  Found {len(manifest)} utterances")

    # Print recognizer info
    print(f"\nRecognizer configuration:")
    print(f"  Encoder: {args.encoder}")
    print(f"  Decoder: {args.decoder}")
    print(f"  Tokens: {args.tokens}")
    if args.num_workers > 1:
        print(f"  Workers: {args.num_workers} (parallel)")
    else:
        print(f"  Workers: 1 (sequential)")

    # Run benchmark
    details_path, summary_path = run_benchmark(
        manifest=manifest,
        data_dir=data_dir,
        output_dir=output_dir,
        encoder=args.encoder,
        decoder=args.decoder,
        tokens=args.tokens,
        language=args.language,
        num_workers=args.num_workers,
    )

    print("\n" + "=" * 60)
    print("Benchmark complete!")
    print("=" * 60)
    print(f"Details: {details_path}")
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
