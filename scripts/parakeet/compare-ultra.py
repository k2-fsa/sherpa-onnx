#!/usr/bin/env python3
"""Reproduce the quiet-audio failure and validate automatic Ultra selection.

Uses a pinned public int8 export (about 670 MB), public speech, and the same
pinned NVIDIA extractor as compare-reference.py. Requires its dependencies plus
onnx==1.20.1. Original downloaded model files are never modified.
"""

import argparse
import importlib.util
import io
import json
from pathlib import Path
import subprocess
import sys
import tarfile
import urllib.request

import numpy as np
import onnx
import soundfile as sf
import torch

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("reference", HERE / "compare-reference.py")
ref = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ref)
REPO = "tpmaerk/parakeet-ultra-sherpa-onnx-int8"
REVISION = "b86784a3a5356cbd126f710b61cd90c5fe5898f1"
CONFIG_REPO = "Olicorne/parakeet-tdt-0.6b-v3-ultra-onnx"
CONFIG_REVISION = "a1fe742758beb8ddc918bff00e1e319fe7e4d3e3"
HASHES = {
    "encoder.int8.onnx": "f4d06712d32431b1718eceba2153c314ac4425e8fe708a3534f5e56c5ad53697",
    "decoder.int8.onnx": "94373c6405c8149d1c9d1e9862514e0e3e88dab1c7de18fac9d5df9c6ba5efdd",
    "joiner.int8.onnx": "8a71aaccfdba3d451775507a889afd2257d2e754463fd81095a8c8f290f924c2",
    "tokens.txt": "d58544679ea4bc6ac563d1f545eb7d474bd6cfa467f0a6e2c1dc1c7d37e3c35d",
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probe", type=Path, required=True)
    parser.add_argument("--native-offline", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    root, original = args.output_dir.resolve(), args.model_dir.resolve()
    if root.exists() or args.threads < 1:
        parser.error("use a new output directory and a positive thread count")
    root.mkdir(parents=True)
    original.mkdir(parents=True, exist_ok=True)
    probe, native = str(args.probe.resolve()), str(args.native_offline.resolve())
    torch.set_num_threads(args.threads)
    for name, expected in HASHES.items():
        ref.fetch(f"https://huggingface.co/{REPO}/resolve/{REVISION}/{name}", original / name, expected)

    # Independently retrieve the converted Ultra config. Equality here does not
    # establish what preprocessing Moondream used during training or in Photon.
    url = (f"https://huggingface.co/{CONFIG_REPO}/resolve/{CONFIG_REVISION}/"
           "parakeet-tdt-0.6b-v3-ultra.nemo")
    request = urllib.request.Request(url, headers={"Range": "bytes=0-2097151"})
    with urllib.request.urlopen(request, timeout=60) as response:
        prefix = response.read(2097152)
    config_path = root / "ultra-model_config.yaml"
    with tarfile.open(fileobj=io.BytesIO(prefix), mode="r|*") as archive:
        for member in archive:
            if member.name.endswith("model_config.yaml"):
                config_path.write_bytes(archive.extractfile(member).read())
                break
    assert ref.sha256(config_path) == ref.CONFIG_SHA256

    # Regenerate features and run all ten existing model-free parity checks.
    fixtures = root / "features"
    with (root / "features.log").open("w") as log:
        subprocess.run([sys.executable, str(HERE / "compare-reference.py"),
                        "--probe", probe, "--output-dir", str(fixtures),
                        "--threads", str(args.threads)], stdout=log, stderr=log, check=True)
    config = ref.yaml.safe_load(config_path.read_text())["preprocessor"]
    extractor = ref.load_nemo(fixtures / "nemo-current.py", config)

    # Use untouched downloads; no metadata labeling or re-export is required.
    raw = root / "model"
    raw.mkdir()
    for part in ("encoder", "decoder", "joiner"):
        (raw / f"{part}.onnx").symlink_to(original / f"{part}.int8.onnx")
    (raw / "tokens.txt").symlink_to(original / "tokens.txt")

    decode, metadata, ort_version = ref.decoder(raw, args.threads)
    rows = []
    passed = True
    for name in ("en", "de", "es", "fr", "en-quiet"):
        samples = np.fromfile(fixtures / f"{name}.f32", dtype="<f4")
        features = {mode: np.fromfile(fixtures / f"{name}-{mode}.f32", dtype="<f4").reshape(-1, 128)
                    for mode in ("legacy", "reference")}
        features["nvidia"] = ref.reference_features(extractor, samples)
        row = dict(fixture=name, pcm_sha256=ref.sha256(fixtures / f"{name}.f32"),
                   decoding={mode: decode(f) for mode, f in features.items()})
        row["same_reference_tokens"] = (
            row["decoding"]["reference"]["token_ids"] == row["decoding"]["nvidia"]["token_ids"])
        passed &= row["same_reference_tokens"]
        wav = root / f"{name}-16k.wav"
        sf.write(wav, samples, 16000, subtype="FLOAT")
        command = [native, "--model-type=nemo_transducer", "--decoding-method=greedy_search",
                   f"--num-threads={args.threads}", "--provider=cpu"]
        command += [f"--{part}={raw / (part + '.onnx')}"
                    for part in ("encoder", "decoder", "joiner")]
        command += [f"--tokens={raw / 'tokens.txt'}", str(wav)]
        run = subprocess.run(command, text=True, capture_output=True, check=True)
        (root / f"{name}-native.log").write_text(run.stderr)
        row["native"] = json.loads(run.stdout)
        passed &= row["native"]["text"] == row["decoding"]["reference"]["text"]
        print(json.dumps(row), flush=True)
        rows.append(row)

    auto = root / "auto.f32"
    subprocess.run([probe, "auto", str(fixtures / "en.f32"), str(auto), str(raw)], check=True)
    selection_exact = auto.read_bytes() == (fixtures / "en-reference.f32").read_bytes()
    passed &= selection_exact

    # Mutated copies are selection tests only; the speech validation above
    # always uses the original export. Exercise the known URL spellings, TDT
    # detection for the plain URL, and unrelated-model fallback.
    cases = root / "metadata-cases"
    cases.mkdir()
    for name in ("decoder.onnx", "joiner.onnx", "tokens.txt"):
        (cases / name).symlink_to(raw / name)
    encoder = onnx.load(original / "encoder.int8.onnx")
    base = "https://huggingface.co/moondream/parakeet-ultra"
    checks = [
        ("v3", "https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3", "reference"),
        ("ultra-plain", base, "reference"),
        ("ultra-tdt", base + " (parakeet-tdt architecture)", "reference"),
        ("ultra-v3", base + " (parakeet-tdt-0.6b-v3 architecture)", "reference"),
        ("ultra-retrained", base + " (retrained nvidia/parakeet-tdt-0.6b-v3)", "reference"),
        ("unrelated", "https://example.com/custom-tdt", "legacy"),
        ("ultra-other", base + "-other-tdt", "legacy"),
        ("parakeet-v2", "https://huggingface.co/nvidia/parakeet-tdt-0.6b-v2", "legacy"),
    ]
    check_results = []
    for name, url, expected in checks:
        onnx.helper.set_model_props(encoder, dict(metadata, url=url))
        onnx.save(encoder, cases / "encoder.onnx")
        output = cases / f"{name}.f32"
        # auto also runs decoder/batch checks; this verifies plain Ultra URLs
        # select TDT despite not containing the historical 'tdt' substring.
        mode = "auto" if name == "ultra-plain" else "select"
        run = subprocess.run([probe, mode, str(fixtures / "en.f32"), str(output),
                              str(cases)], text=True, capture_output=True)
        (cases / f"{name}.log").write_text(run.stderr)
        ok = (run.returncode == 0
              and output.read_bytes() == (fixtures / f"en-{expected}.f32").read_bytes())
        check_results.append(dict(case=name, passed=ok, returncode=run.returncode))
        passed &= ok
    (cases / "encoder.onnx").unlink()
    quiet = next(row for row in rows if row["fixture"] == "en-quiet")
    reproduced = (quiet["decoding"]["legacy"]["text"] == ""
                  and quiet["decoding"]["reference"]["text"] ==
                  "Ask not what your country can do for you. Ask what you can do for your country.")
    passed &= reproduced
    report = dict(passed=bool(passed), model_repo=REPO, model_revision=REVISION,
                  config_repo=CONFIG_REPO, config_revision=CONFIG_REVISION,
                  config_sha256=ref.CONFIG_SHA256, model_hashes=HASHES,
                  metadata=metadata, original_model_files_unchanged=True,
                  onnx=onnx.__version__, onnxruntime=ort_version,
                  feature_report="features/results.json", results=rows,
                  quiet_failure_reproduced=reproduced, selection_exact=selection_exact,
                  short_only_and_mixed_batches_pass=True, metadata_checks=check_results)
    (root / "results.json").write_text(json.dumps(report, indent=2) + "\n")
    if not passed:
        raise SystemExit("Ultra validation failed; inspect results.json")
    print("Ultra validation passed", flush=True)


if __name__ == "__main__":
    main()
