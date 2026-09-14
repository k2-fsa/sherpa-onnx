# Orukeet with the offline WebSocket server

[Orukeet](https://huggingface.co/oruk/orukeet) is a 25-language fine-tune of
Parakeet TDT 0.6B v3. Its ONNX export uses sherpa-onnx's existing NeMo transducer
implementation. The fitted Gabor filters are stored as convolution weights;
inference requires no custom operators.

## Download

The following commands download the release's manifest and INT8 model archive
from an immutable Hugging Face revision. The manifest supplies the archive's
size, checksum, and extraction directory.

```bash
set -e
base=https://huggingface.co/oruk/orukeet/resolve/55a984d46f68323301837194ce647c702f55facc/onnx
curl -fL "$base/manifest.json" -o orukeet-manifest.json
curl -fL "$base/sherpa-onnx-orukeet-v0.1.0-int8.tar.bz2" \
  -o sherpa-onnx-orukeet-v0.1.0-int8.tar.bz2

python3 - <<'PY'
import hashlib
import json
from pathlib import Path

manifest_path = Path("orukeet-manifest.json")
expected = "7e80f93f0e9b923c392424b0f85d28a717feee0a4d2a6aa9bfa723693868e727"
assert hashlib.sha256(manifest_path.read_bytes()).hexdigest() == expected
manifest = json.loads(manifest_path.read_text())
archive = Path(manifest["archive"])
assert archive.stat().st_size == manifest["archive_bytes"]
digest = hashlib.sha256()
with archive.open("rb") as stream:
    for block in iter(lambda: stream.read(1024 * 1024), b""):
        digest.update(block)
assert digest.hexdigest() == manifest["archive_sha256"]
print("Verified:", archive)
PY

tar xf sherpa-onnx-orukeet-v0.1.0-int8.tar.bz2
```

The archive contains `encoder.int8.onnx`, `decoder.int8.onnx`,
`joiner.int8.onnx`, `tokens.txt`, and the weight license and attribution.
Keep these files together. Downloads occupy approximately 487 MB; the extracted
model occupies approximately 672 MB. Existing model files can be reused offline.

## Serve

From the sherpa-onnx repository root, install the example's dependencies. This
example uses the legacy WebSocket server API:

```bash
python3 -m pip install sherpa-onnx numpy 'websockets<14'
```

Start the server:

```bash
model=./sherpa-onnx-orukeet-v0.1.0-int8
python3 python-api-examples/non_streaming_server.py \
  --model-type nemo_transducer \
  --encoder "$model/encoder.int8.onnx" \
  --decoder "$model/decoder.int8.onnx" \
  --joiner "$model/joiner.int8.onnx" \
  --tokens "$model/tokens.txt" \
  --feat-dim 128 \
  --num-threads 2 \
  --provider cpu \
  --port 6006
```

Transcribe files with the existing client:

```bash
python3 python-api-examples/offline-websocket-client-decode-files-sequential.py \
  --server-addr localhost --server-port 6006 recording.wav
```

Audio is decoded locally. The model uses a 16 kHz frontend and the existing
greedy TDT decoder. This is utterance-based recognition: the client sends each
recording for a final transcript.

For microphone input, download `silero_vad.onnx` as described in
[vad-with-non-streaming-asr.py](./vad-with-non-streaming-asr.py), then run:

```bash
python3 -m pip install sounddevice
model=./sherpa-onnx-orukeet-v0.1.0-int8
python3 python-api-examples/vad-with-non-streaming-asr.py \
  --silero-vad-model ./silero_vad.onnx \
  --model-type nemo_transducer \
  --encoder "$model/encoder.int8.onnx" \
  --decoder "$model/decoder.int8.onnx" \
  --joiner "$model/joiner.int8.onnx" \
  --tokens "$model/tokens.txt" \
  --feature-dim 128 --num-threads 2
```

Stock Parakeet TDT v3 uses the same server options with its own model directory.
Other transducer examples retain the default `--model-type transducer`.
