# GigaAM-v4 CTC E2E

[GigaAM-v4](https://github.com/salute-developers/GigaAM) recognizes Russian and
English. The [INT8 CTC export](https://huggingface.co/ai-babai/gigaam-v4-onnx/tree/v0.2.0/ctc-e2e)
produces punctuated text with the existing offline NeMo CTC loader and the
built-in GigaAM frontend.

## Download the model

From the sherpa-onnx repository root, download the model (242 MB) and token table.
The example uses `curl` and `shasum` on macOS and Linux.

```bash
(
  set -e
  mkdir -p gigaam-v4-ctc-e2e && cd gigaam-v4-ctc-e2e
  base=https://huggingface.co/ai-babai/gigaam-v4-onnx/resolve/d39e2a200f596315acef7003d246b65a73e9a327/ctc-e2e
  curl -fLO "$base/model.int8.onnx"
  curl -fLO "$base/tokens.txt"
  shasum -a 256 -c <<'EOF'
0f61ec8ca69517cb68e24531c5000053b80ab84cf7071cc294571edfae493462  model.int8.onnx
c6a37fab667d53c37e66835c90083808daa116c3993c5e93ce518b5a3896de6e  tokens.txt
EOF
)
```

## Transcribe a WAV file

Use a Python 3.11 or 3.12 virtual environment in the repository root.
Replace `recording.wav` with a 16 kHz mono, 16-bit PCM WAV containing one
utterance (20 ms to 30 s). Segment continuous audio at speech boundaries.
A silent clip can produce text. Pass one file per invocation for reproducible
single-clip decoding. Batching clips of different lengths can change the text.

```bash
python3 -m pip install sherpa-onnx==2.0.3 numpy==2.4.6
python3 python-api-examples/offline-decode-files.py \
  --nemo-ctc=gigaam-v4-ctc-e2e/model.int8.onnx \
  --tokens=gigaam-v4-ctc-e2e/tokens.txt \
  --sample-rate=16000 --feature-dim=64 --num-threads=3 \
  recording.wav
```
