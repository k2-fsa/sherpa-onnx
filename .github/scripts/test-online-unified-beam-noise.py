#!/usr/bin/env python3
"""Check the released Parakeet Unified 1120 ms INT8 model on nonspeech audio."""

import argparse
import json
import os
import random
import subprocess
import sys
import tempfile
import wave
from array import array
from pathlib import Path


def write_fixtures(directory: Path) -> tuple[Path, Path]:
    sample_rate = 16000
    num_samples = 16 * sample_rate
    rng = random.Random(4131)
    noise_rms = 32767 * 10 ** (-45 / 20)
    noise = array("h", (round(rng.gauss(0, noise_rms)) for _ in range(num_samples)))
    if sys.byteorder == "big":
        noise.byteswap()

    silence_path = directory / "silence.wav"
    noise_path = directory / "white-noise.wav"
    for path, samples in (
        (silence_path, bytes(num_samples * 2)),
        (noise_path, noise.tobytes()),
    ):
        with wave.open(str(path), "wb") as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(sample_rate)
            wav.writeframes(samples)
    return silence_path, noise_path


def check_results(output: str, fixtures: tuple[Path, Path]) -> None:
    # With endpointing disabled the CLI emits one JSON result per input WAV.
    results = [
        json.loads(line)
        for line in output.splitlines()
        if line.lstrip().startswith("{")
    ]
    if len(results) != len(fixtures):
        raise ValueError(
            f"Expected {len(fixtures)} recognition results, received {len(results)}"
        )
    for fixture, result in zip(fixtures, results):
        if (
            not isinstance(result, dict)
            or result.get("tokens") != []
            or result.get("text") != ""
        ):
            raise ValueError(f"Expected zero tokens and empty text for {fixture.name}")
        trailing_blanks = result.get("num_trailing_blanks")
        if type(trailing_blanks) is not int or trailing_blanks <= 0:
            raise ValueError(f"No blank frames decoded for {fixture.name}")
        print(f"{fixture.name}: zero tokens, {trailing_blanks} trailing blank frames")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--exe", required=True, type=Path, help="Path to the built sherpa-onnx CLI"
    )
    parser.add_argument(
        "--model-dir", required=True, type=Path, help="Extracted 1120 ms INT8 model"
    )
    args = parser.parse_args()

    try:
        exe = args.exe.resolve(strict=True)
        model_dir = args.model_dir.resolve(strict=True)
        if not exe.is_file() or not os.access(exe, os.X_OK):
            raise ValueError("--exe must be an executable file")
        if not model_dir.is_dir():
            raise ValueError("--model-dir must be a directory")
        command = [
            str(exe),
            "--provider=cpu",
            "--num-threads=2",
            "--enable-endpoint=false",
            "--left-padding=0",
            "--right-padding=0",
            "--decoding-method=modified_beam_search",
            "--max-active-paths=8",
            "--blank-penalty=0.25",
        ]
        for flag, filename in (
            ("encoder", "encoder.int8.onnx"),
            ("decoder", "decoder.int8.onnx"),
            ("joiner", "joiner.int8.onnx"),
            ("tokens", "tokens.txt"),
        ):
            path = model_dir / filename
            if path.is_symlink() or not path.is_file() or path.stat().st_size == 0:
                raise ValueError(f"Expected a nonempty model file named {filename}")
            command.append(f"--{flag}={path}")

        with tempfile.TemporaryDirectory() as directory:
            fixtures = write_fixtures(Path(directory))
            result = subprocess.run(
                [*command, *(str(path) for path in fixtures)],
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                encoding="utf-8",
                timeout=180,
            )
            check_results(result.stdout, fixtures)
    except (OSError, subprocess.SubprocessError, ValueError) as exc:
        parser.exit(status=1, message=f"Unified beam smoke failed: {exc}\n")


if __name__ == "__main__":
    main()
