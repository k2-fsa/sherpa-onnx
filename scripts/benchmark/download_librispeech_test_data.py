#!/usr/bin/env python3
# /// script
# dependencies = ["gdown"]
# ///
from __future__ import annotations

"""
Download and prepare LibriSpeech test data for timestamp benchmarking.

Downloads:
1. LibriSpeech dev-clean audio subset
2. MFA word alignments from librispeech-alignments repo

Outputs:
- benchmark_data/audio/*.wav (16kHz mono WAV files)
- benchmark_data/manifest.json (mapping of audio files to ground truth timestamps)

Usage:
    python scripts/benchmark/download_librispeech_test_data.py [--num-utterances 200]
"""

import argparse
import json
import os
import re
import subprocess
import sys
import tarfile
import tempfile
import urllib.request
import zipfile
from pathlib import Path

# URLs for downloads
LIBRISPEECH_DEV_CLEAN_URL = "https://www.openslr.org/resources/12/dev-clean.tar.gz"
MFA_ALIGNMENTS_URL = "https://drive.google.com/uc?export=download&id=1WYfgr31T-PPwMcxuAq09XZfHQO5Mw8fE"

# Google Drive file ID for the simple TXT format alignments
GDRIVE_FILE_ID = "1WYfgr31T-PPwMcxuAq09XZfHQO5Mw8fE"


def download_file(url: str, dest_path: Path, description: str = "file"):
    """Download a file with progress indication."""
    print(f"Downloading {description}...")
    print(f"  URL: {url}")
    print(f"  Destination: {dest_path}")

    def reporthook(block_num, block_size, total_size):
        if total_size > 0:
            downloaded = block_num * block_size
            percent = min(100, downloaded * 100 / total_size)
            mb_downloaded = downloaded / (1024 * 1024)
            mb_total = total_size / (1024 * 1024)
            sys.stdout.write(f"\r  Progress: {percent:.1f}% ({mb_downloaded:.1f}/{mb_total:.1f} MB)")
            sys.stdout.flush()

    urllib.request.urlretrieve(url, dest_path, reporthook)
    print()  # newline after progress


def download_from_gdrive(file_id: str, dest_path: Path, description: str = "file"):
    """Download a file from Google Drive using gdown."""
    try:
        import gdown
    except ImportError:
        print("ERROR: gdown is required for downloading from Google Drive.")
        print("Install it with: pip install gdown")
        sys.exit(1)

    print(f"Downloading {description} from Google Drive...")
    print(f"  File ID: {file_id}")
    print(f"  Destination: {dest_path}")

    url = f"https://drive.google.com/uc?id={file_id}"
    gdown.download(url, str(dest_path), quiet=False)


def extract_tar_gz(archive_path: Path, dest_dir: Path):
    """Extract a tar.gz archive."""
    print(f"Extracting {archive_path}...")
    with tarfile.open(archive_path, "r:gz") as tar:
        tar.extractall(dest_dir)
    print(f"  Extracted to {dest_dir}")


def extract_zip(archive_path: Path, dest_dir: Path):
    """Extract a zip archive."""
    print(f"Extracting {archive_path}...")
    with zipfile.ZipFile(archive_path, "r") as z:
        z.extractall(dest_dir)
    print(f"  Extracted to {dest_dir}")


def convert_flac_to_wav(flac_path: Path, wav_path: Path):
    """Convert FLAC to 16kHz mono WAV using ffmpeg or sox."""
    wav_path.parent.mkdir(parents=True, exist_ok=True)

    # Try ffmpeg first
    try:
        subprocess.run(
            [
                "ffmpeg", "-y", "-i", str(flac_path),
                "-ar", "16000", "-ac", "1", "-sample_fmt", "s16",
                str(wav_path)
            ],
            check=True,
            capture_output=True
        )
        return
    except (subprocess.CalledProcessError, FileNotFoundError):
        pass

    # Try sox
    try:
        subprocess.run(
            ["sox", str(flac_path), "-r", "16000", "-c", "1", str(wav_path)],
            check=True,
            capture_output=True
        )
        return
    except (subprocess.CalledProcessError, FileNotFoundError):
        pass

    print(f"ERROR: Could not convert {flac_path}")
    print("Please install ffmpeg or sox")
    sys.exit(1)


def parse_alignment_line(line: str) -> dict | None:
    """
    Parse a single line from the MFA alignment file.

    Format: utterance_id "word1,word2,..." "end_time1,end_time2,..."
    Empty words represent silences.
    Times are END times for each word.

    Returns dict with utterance_id, words (list), and word_times (list of {word, start, end})
    """
    # Pattern: utterance_id "words" "times"
    match = re.match(r'^(\S+)\s+"([^"]*)"\s+"([^"]*)"', line.strip())
    if not match:
        return None

    utterance_id = match.group(1)
    words_str = match.group(2)
    times_str = match.group(3)

    # Parse words (comma-separated, may have empty entries for silences)
    words = words_str.split(",")

    # Parse end times
    try:
        end_times = [float(t) for t in times_str.split(",") if t]
    except ValueError:
        return None

    if len(words) != len(end_times):
        return None

    # Convert to word_times with start and end
    word_times = []
    prev_end = 0.0
    for word, end_time in zip(words, end_times):
        if word:  # Skip empty words (silences)
            word_times.append({
                "word": word,
                "start": prev_end,
                "end": end_time
            })
        prev_end = end_time

    return {
        "utterance_id": utterance_id,
        "words": [w["word"] for w in word_times],
        "word_times": word_times
    }


def parse_alignment_file(alignment_path: Path) -> dict:
    """Parse an alignment file and return dict mapping utterance_id to alignment data."""
    alignments = {}
    with open(alignment_path, "r") as f:
        for line in f:
            parsed = parse_alignment_line(line)
            if parsed:
                alignments[parsed["utterance_id"]] = parsed
    return alignments


def main():
    parser = argparse.ArgumentParser(description="Download LibriSpeech benchmark data")
    parser.add_argument(
        "--num-utterances",
        type=int,
        default=200,
        help="Number of utterances to include in test set (default: 200)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="benchmark_data",
        help="Output directory (default: benchmark_data)"
    )
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Skip download step (use existing files)"
    )
    args = parser.parse_args()

    # Paths
    script_dir = Path(__file__).parent.resolve()
    repo_root = script_dir.parent.parent
    output_dir = repo_root / args.output_dir
    audio_dir = output_dir / "audio"
    cache_dir = output_dir / ".cache"

    output_dir.mkdir(parents=True, exist_ok=True)
    audio_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: Download LibriSpeech dev-clean
    librispeech_tar = cache_dir / "dev-clean.tar.gz"
    librispeech_dir = cache_dir / "LibriSpeech" / "dev-clean"

    if not args.skip_download and not librispeech_dir.exists():
        if not librispeech_tar.exists():
            download_file(
                LIBRISPEECH_DEV_CLEAN_URL,
                librispeech_tar,
                "LibriSpeech dev-clean"
            )
        extract_tar_gz(librispeech_tar, cache_dir)

    # Step 2: Download MFA alignments
    alignments_zip = cache_dir / "librispeech-alignments.zip"
    alignments_dir = cache_dir / "alignments"

    if not args.skip_download and not alignments_dir.exists():
        if not alignments_zip.exists():
            download_from_gdrive(
                GDRIVE_FILE_ID,
                alignments_zip,
                "LibriSpeech MFA alignments"
            )
        alignments_dir.mkdir(parents=True, exist_ok=True)
        extract_zip(alignments_zip, alignments_dir)

    # Step 3: Find alignment files for dev-clean
    print("\nParsing alignment files...")
    all_alignments = {}

    # Look for dev-clean alignment files
    for alignment_file in alignments_dir.rglob("*.alignment.txt"):
        if "dev-clean" in str(alignment_file):
            print(f"  Parsing {alignment_file.name}...")
            file_alignments = parse_alignment_file(alignment_file)
            all_alignments.update(file_alignments)

    print(f"  Found {len(all_alignments)} alignments")

    # Step 4: Find corresponding audio files and convert
    print("\nProcessing audio files...")
    manifest = []
    processed = 0

    # Walk through LibriSpeech directory structure: speaker/chapter/utterance.flac
    for flac_file in sorted(librispeech_dir.rglob("*.flac")):
        utterance_id = flac_file.stem  # e.g., "84-121123-0000"

        if utterance_id not in all_alignments:
            continue

        alignment = all_alignments[utterance_id]

        # Skip utterances with no words
        if not alignment["word_times"]:
            continue

        # Convert to WAV
        wav_file = audio_dir / f"{utterance_id}.wav"
        if not wav_file.exists():
            convert_flac_to_wav(flac_file, wav_file)

        # Add to manifest
        manifest.append({
            "utterance_id": utterance_id,
            "audio_path": str(wav_file.relative_to(output_dir)),
            "transcript": " ".join(alignment["words"]),
            "word_times": alignment["word_times"]
        })

        processed += 1
        if processed % 50 == 0:
            print(f"  Processed {processed} utterances...")

        if processed >= args.num_utterances:
            break

    print(f"\nProcessed {len(manifest)} utterances")

    # Step 5: Write manifest
    manifest_path = output_dir / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"Wrote manifest to {manifest_path}")

    # Summary
    print("\n" + "=" * 60)
    print("Download complete!")
    print("=" * 60)
    print(f"Audio files: {audio_dir}")
    print(f"Manifest: {manifest_path}")
    print(f"Total utterances: {len(manifest)}")

    # Calculate total duration
    total_words = sum(len(item["word_times"]) for item in manifest)
    print(f"Total words: {total_words}")


if __name__ == "__main__":
    main()
