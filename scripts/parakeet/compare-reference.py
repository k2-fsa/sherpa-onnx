#!/usr/bin/env python3
"""Compare the native frontend with pinned NVIDIA code and public recordings.

Requires numpy, torch (CPU is sufficient), librosa, pyyaml, and optionally
onnxruntime for decoding. Downloads reference source, checkpoint configuration,
and four small public WAVs. --download-model additionally downloads the FP32
ONNX export (about 2.4 GB). No private audio is accepted by this script.
"""

import argparse
import ast
import hashlib
import io
import json
import logging
import math
from pathlib import Path
import platform
import random
import subprocess
import tarfile
import urllib.request

import librosa
import numpy as np
import soundfile as sf
import torch
import yaml

NEMO_CURRENT = "50c71dbe1534a89e0d07fe66fbffd205408062ad"
NEMO_23 = "2b03b748bb7d28e4e382df3d085078c6146d174e"
CHECKPOINT = "541d1f99c6b0c3cd0b11a95167540bb8edefd82b"
EXPORT = "1a468a35cbba69418f126de829e75261dea4a4e4"
EXPORT_REPO = "csukuangfj/sherpa-onnx-nemo-parakeet-tdt-0.6b-v3"
CURRENT_SHA256 = "5cbc7b5eff0c2e57c4d9e548e5abc33b8c20078e48661faba9f5872c123d4321"
HISTORICAL_SHA256 = "bd333719eec5b65053a97e29f12618c9e82121fb0ab0c1c8854bb47a1114b101"
CONFIG_SHA256 = "eadae985ef97451402be2faef3b240c2cae7d9cc0362c5bd00e64dfc4265ed09"


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def fetch(url, path, expected=None):
    if not path.exists():
        temporary = path.with_name(path.name + ".download")
        with urllib.request.urlopen(url, timeout=60) as src, temporary.open("wb") as dst:
            for block in iter(lambda: src.read(1024 * 1024), b""):
                dst.write(block)
        temporary.replace(path)
    actual = sha256(path)
    if expected and actual != expected:
        raise ValueError(f"checksum mismatch: {path}")
    return actual


def load_nemo(path, config):
    # Execute NVIDIA's original AST nodes, without rewriting any operations.
    # This avoids importing the unrelated NeMo training/model/augmentation stack.
    # forward_packed is not used; its annotation alone needs the placeholder.
    tree = ast.parse(path.read_text())
    names = {"normalize_batch", "splice_frames", "FilterbankFeatures"}
    nodes = [n for n in tree.body if getattr(n, "name", None) in names
             or isinstance(n, ast.Assign) and any(
                 isinstance(t, ast.Name) and t.id == "CONSTANT" for t in n.targets)]
    namespace = dict(torch=torch, nn=torch.nn, np=np, librosa=librosa,
                     math=math, random=random, logging=logging,
                     PackedEncoderActivations=object)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), namespace)
    return namespace["FilterbankFeatures"](
        sample_rate=config["sample_rate"],
        n_window_size=int(config["window_size"] * config["sample_rate"]),
        n_window_stride=int(config["window_stride"] * config["sample_rate"]),
        window=config["window"], normalize=config["normalize"],
        n_fft=config["n_fft"], nfilt=config["features"], log=config["log"],
        frame_splicing=config["frame_splicing"], dither=config["dither"],
        pad_to=config["pad_to"], pad_value=config["pad_value"],
    ).eval()


def reference_features(extractor, samples):
    with torch.no_grad():
        features, length = extractor(torch.from_numpy(samples.copy())[None],
                                     torch.tensor([len(samples)]))
    return features[0, :, :int(length[0])].T.contiguous().numpy()


def double_normalized_features(extractor, samples):
    # Diagnostic only: NVIDIA's unchanged STFT/mel/log, followed by float64
    # mean/sample-variance math. The primary reference remains NVIDIA forward.
    normalize = extractor.normalize
    try:
        extractor.normalize = None
        raw = reference_features(extractor, samples).astype(np.float64)
    finally:
        extractor.normalize = normalize
    return (raw - raw.mean(axis=0)) / (raw.std(axis=0, ddof=1) + 1e-5)


def metrics(actual, reference):
    n = min(len(actual), len(reference))
    delta = actual[:n].astype(np.float64) - reference[:n].astype(np.float64)
    return dict(frames=len(actual), reference_frames=len(reference),
                same_length=len(actual) == len(reference),
                rmse=float(np.sqrt(np.mean(delta * delta))) if n else None,
                max_abs=float(np.max(np.abs(delta))) if n else None,
                finite=bool(np.isfinite(actual).all()))


def decoder(model_dir, threads):
    import onnxruntime as ort

    options = ort.SessionOptions()
    options.intra_op_num_threads = threads
    options.inter_op_num_threads = 1
    sessions = {name: ort.InferenceSession(str(model_dir / f"{name}.onnx"),
                sess_options=options, providers=["CPUExecutionProvider"])
                for name in ("encoder", "decoder", "joiner")}
    meta = sessions["encoder"].get_modelmeta().custom_metadata_map
    assert meta["url"] == "https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3"
    assert meta["feat_dim"] == "128" and meta["normalize_type"] == "per_feature"
    blank = int(meta["vocab_size"])
    tokens = {int(line.rsplit(" ", 1)[1]): line.rsplit(" ", 1)[0]
              for line in (model_dir / "tokens.txt").read_text().splitlines()}

    def run(name, values):
        session = sessions[name]
        return session.run(None, dict(zip([v.name for v in session.get_inputs()], values)))

    def decode(features):
        enc, length = run("encoder", [features.T[None].copy(), np.array([len(features)], np.int64)])
        shape = (int(meta["pred_rnn_layers"]), 1, int(meta["pred_hidden"]))

        def predict(token, states):
            return run("decoder", [np.array([[token]], np.int32), np.array([1], np.int32)] + states)

        dec = predict(blank, [np.zeros(shape, np.float32) for _ in range(2)])
        ids, frame, emitted, steps = [], 0, 0, 0
        while frame < int(length[0]):
            steps += 1
            if steps > int(length[0]) * 6 + 10:
                raise RuntimeError("greedy TDT decoder did not advance")
            logits = run("joiner", [enc[:, :, frame:frame + 1], dec[0]])[0].reshape(-1)
            assert len(logits) == blank + 1 + 5  # durations [0, 1, 2, 3, 4]
            token, skip = int(logits[:blank + 1].argmax()), int(logits[blank + 1:].argmax())
            if token != blank:
                ids.append(token)
                dec = predict(token, dec[2:])
                emitted += 1
            if skip > 0:
                emitted = 0
            if emitted >= 5 or (token == blank and skip == 0):
                emitted, skip = 0, 1
            frame += skip
        return dict(text="".join(tokens[i] for i in ids).replace("▁", " ").strip(), token_ids=ids)

    return decode, meta, ort.__version__


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probe", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path)
    parser.add_argument("--download-model", action="store_true")
    parser.add_argument("--native-offline", type=Path)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    if args.threads < 1 or args.download_model and args.model_dir is None:
        parser.error("positive threads and --model-dir for --download-model are required")
    if args.native_offline and args.model_dir is None:
        parser.error("--native-offline requires --model-dir")
    root = args.output_dir.resolve()
    root.mkdir(parents=True, exist_ok=True)
    if (root / "results.json").exists():
        parser.error("results.json exists; choose a new output directory")
    torch.set_num_threads(args.threads)
    manifest = dict(nemo_current=NEMO_CURRENT, nemo_historical=NEMO_23,
                    checkpoint=CHECKPOINT, onnx_export=EXPORT,
                    torch=torch.__version__, numpy=np.__version__, librosa=librosa.__version__, hashes={})
    manifest["feature_tolerances"] = dict(
        nvidia_float32=dict(rmse=3e-4, max_abs=3e-3),
        double_normalization_diagnostic=dict(rmse=1e-4, max_abs=1e-3))
    manifest["python"] = platform.python_version()
    manifest["platform"] = platform.platform()
    manifest["native_onnxruntime"] = subprocess.check_output(
        [str(args.probe.resolve()), "--version"], text=True).strip()
    extractors = {}
    for name, rev in [("current", NEMO_CURRENT), ("historical", NEMO_23)]:
        path = root / f"nemo-{name}.py"
        url = (f"https://raw.githubusercontent.com/NVIDIA/NeMo/{rev}/"
               "nemo/collections/asr/parts/preprocessing/features.py")
        expected = CURRENT_SHA256 if name == "current" else HISTORICAL_SHA256
        manifest["hashes"][path.name] = fetch(url, path, expected)
    config_path = root / "model_config.yaml"
    if not config_path.exists():
        url = f"https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3/resolve/{CHECKPOINT}/parakeet-tdt-0.6b-v3.nemo"
        # The config is near the beginning of this uncompressed tar checkpoint.
        with urllib.request.urlopen(urllib.request.Request(url, headers={"Range": "bytes=0-1048575"})) as src:
            prefix = src.read(1048576)
        with tarfile.open(fileobj=io.BytesIO(prefix), mode="r|*") as archive:
            for member in archive:
                if member.name.endswith("model_config.yaml"):
                    config_path.write_bytes(archive.extractfile(member).read())
                    break
    if sha256(config_path) != CONFIG_SHA256:
        raise ValueError("checkpoint config checksum mismatch")
    config = yaml.safe_load(config_path.read_text())
    manifest["checkpoint_nemo_version"] = config["nemo_version"]
    manifest["preprocessor"] = config["preprocessor"]
    manifest["hashes"][config_path.name] = CONFIG_SHA256
    for name in ("current", "historical"):
        extractors[name] = load_nemo(root / f"nemo-{name}.py", config["preprocessor"])

    tree_url = f"https://huggingface.co/api/models/{EXPORT_REPO}/tree/{EXPORT}?recursive=true"
    tree_path = root / "export-tree.json"
    manifest["hashes"][tree_path.name] = fetch(tree_url, tree_path)
    tree = {item["path"]: item for item in json.loads(tree_path.read_text())}
    samples_by_name, wavs = {}, {}
    for lang in ("en", "de", "es", "fr"):
        name = f"test_wavs/{lang}.wav"
        path = root / f"{lang}.wav"
        manifest["hashes"][path.name] = fetch(
            f"https://huggingface.co/{EXPORT_REPO}/resolve/{EXPORT}/{name}", path, tree[name]["lfs"]["oid"])
        samples, rate = sf.read(path, dtype="float32", always_2d=True)
        assert samples.shape[1] == 1
        samples = samples[:, 0]
        if rate != 16000:
            input_path, output_path = root / f"{lang}-original.f32", root / f"{lang}-resampled.f32"
            samples.astype("<f4").tofile(input_path)
            subprocess.run([str(args.probe.resolve()), "resample", str(input_path),
                            str(output_path), str(rate)], check=True)
            samples = np.fromfile(output_path, dtype="<f4")
        samples_by_name[lang] = samples
        wavs[lang] = path
    samples_by_name["en-quiet"] = samples_by_name["en"] * np.float32(0.001)
    # Preserve the exact attenuated float32 input for the native CLI too.
    sf.write(root / "en-quiet.wav", samples_by_name["en-quiet"], 16000, subtype="FLOAT")
    wavs["en-quiet"] = root / "en-quiet.wav"
    for n in (320, 321, 1281, 16001):
        state, values = 42, []
        for _ in range(n):
            state = (state * 1664525 + 1013904223) & 0xffffffff
            values.append(((state >> 16) - 32768) / 327680.0)
        samples_by_name[f"broadband-{n}"] = np.array(values, dtype=np.float32)
    samples_by_name["silence"] = np.zeros(16000, np.float32)

    decode = None
    if args.model_dir:
        model_dir = args.model_dir.resolve()
        model_dir.mkdir(parents=True, exist_ok=True)
        for name in ("encoder.onnx", "encoder.weights", "decoder.onnx", "joiner.onnx", "tokens.txt"):
            path = model_dir / name
            if args.download_model:
                fetch(f"https://huggingface.co/{EXPORT_REPO}/resolve/{EXPORT}/{name}", path)
            actual = sha256(path)
            expected = tree[name].get("lfs", {}).get("oid")
            if expected and actual != expected:
                raise ValueError(f"ONNX export checksum mismatch: {name}")
            if not expected:
                contents = path.read_bytes()
                blob = hashlib.sha1(f"blob {len(contents)}\0".encode() + contents).hexdigest()
                if blob != tree[name]["oid"]:
                    raise ValueError(f"ONNX export Git blob mismatch: {name}")
            manifest["hashes"][name] = actual
        decode, manifest["model_metadata"], manifest["onnxruntime"] = decoder(model_dir, args.threads)

    results, failed = [], False
    for name, samples in samples_by_name.items():
        input_path = root / f"{name}.f32"
        samples.astype("<f4").tofile(input_path)
        features = {}
        for mode in ("reference", "legacy"):
            path = root / f"{name}-{mode}.f32"
            subprocess.run([str(args.probe.resolve()), mode, str(input_path), str(path)], check=True)
            features[mode] = np.fromfile(path, dtype="<f4").reshape(-1, 128)
        for mode, extractor in extractors.items():
            features[mode] = reference_features(extractor, samples)
        double_reference = double_normalized_features(extractors["current"], samples)
        row = dict(fixture=name, samples=len(samples), pcm_sha256=sha256(input_path),
                   corrected_vs_current=metrics(features["reference"], features["current"]),
                   corrected_vs_double_normalization=metrics(features["reference"], double_reference),
                   nvidia_float32_vs_double_normalization=metrics(features["current"], double_reference),
                   legacy_vs_current=metrics(features["legacy"], features["current"]),
                   corrected_vs_historical=metrics(features["reference"], features["historical"]))
        m = row["corrected_vs_current"]
        row["parity_pass"] = m["same_length"] and m["finite"] and m["rmse"] < 3e-4 and m["max_abs"] < 3e-3
        m_double = row["corrected_vs_double_normalization"]
        row["double_diagnostic_pass"] = (
            m_double["same_length"] and m_double["finite"]
            and m_double["rmse"] < 1e-4 and m_double["max_abs"] < 1e-3)
        failed |= not row["parity_pass"]
        failed |= not row["double_diagnostic_pass"]
        print(f"Feature comparison {name}: {json.dumps(m)}", flush=True)
        if decode and name in wavs:
            row["decoding"] = {mode: decode(features[mode]) for mode in ("reference", "current", "legacy")}
            row["same_reference_tokens"] = (
                row["decoding"]["reference"]["token_ids"] == row["decoding"]["current"]["token_ids"])
            failed |= not row["same_reference_tokens"]
            if name == "en":
                row["decoding"]["historical"] = decode(features["historical"])
                path = root / "en-auto.f32"
                subprocess.run([str(args.probe.resolve()), "auto", str(input_path), str(path), str(model_dir)],
                               check=True)
                row["metadata_selection_exact"] = bool(np.array_equal(
                    np.fromfile(path, dtype="<f4").reshape(-1, 128), features["reference"]))
                row["short_only_and_mixed_batches_pass"] = True
                failed |= not row["metadata_selection_exact"]
            if args.native_offline:
                command = [str(args.native_offline.resolve()), "--model-type=nemo_transducer",
                           "--decoding-method=greedy_search", f"--num-threads={args.threads}", "--provider=cpu"]
                command += [f"--{part}={model_dir / (part + '.onnx')}" for part in ("encoder", "decoder", "joiner")]
                command += [f"--tokens={model_dir / 'tokens.txt'}", str(wavs[name])]
                run = subprocess.run(command, check=True, text=True, capture_output=True)
                (root / f"{name}-native.log").write_text(run.stderr)
                row["native_result"] = json.loads(run.stdout)
                row["native_text_matches"] = row["native_result"]["text"] == row["decoding"]["current"]["text"]
                failed |= not row["native_text_matches"]
        results.append(row)
        print(json.dumps(row), flush=True)
    report = dict(manifest=manifest, results=results, passed=not failed)
    (root / "results.json").write_text(json.dumps(report, indent=2) + "\n")
    if failed:
        raise SystemExit("comparison failed; inspect results.json")


if __name__ == "__main__":
    main()
