# GigaAM offline RNN-T

Use a sherpa-onnx Python build that includes `OfflineRecognizer.from_gigaam`.
Tested with sherpa-onnx 2.0.3 built from source, ONNX Runtime 1.28.2,
and Python 3.12 on macOS arm64.

| Package | Encoder | Language |
|---|---|---|
| [GigaAM-v4 RNN-T v0.2.0](https://huggingface.co/ai-babai/gigaam-v4-onnx/tree/v0.2.0/rnnt) | `v4_rnnt_encoder.int8.onnx` | Russian and English, without a language hint |
| [Multilingual-v2 RNN-T v0.2.0](https://huggingface.co/ai-babai/gigaam-multilingual-v2-onnx/tree/v0.2.0) | `multilingual_v2_rnnt_encoder.int8.onnx` | `auto` or a model language code |
| [Multilingual-v2 RNN-T INT4](https://huggingface.co/ai-babai/gigaam-multilingual-v2-onnx/tree/edfc6b5081b7ed589517cf4dbb793e4dd637b6d3/int4) | `multilingual_v2_rnnt_encoder.int4.onnx` | `auto` or a model language code |

Download the encoder, matching `*_advance.onnx` and one SentencePiece
`*_tokenizer.model` from the same package. Use `head="e2e"` with the e2e
tokenizer for formatted text or `head="raw"` with the raw tokenizer for
plain text. The recognizer runs on CPU and accepts one utterance of up to
30 seconds per stream.

```python
import wave
from pathlib import Path

import numpy as np
import sherpa_onnx

package = Path("multilingual-v2")
recognizer = sherpa_onnx.OfflineRecognizer.from_gigaam(
    encoder=str(package / "multilingual_v2_rnnt_encoder.int8.onnx"),
    advance=str(package / "multilingual_v2_rnnt_advance.onnx"),
    tokenizer=str(package / "multilingual_v2_rnnt_e2e_tokenizer.model"),
    language="auto",
    head="e2e",
    num_threads=3,
)

with wave.open("clip.wav", "rb") as wav:
    assert wav.getnchannels() == 1 and wav.getsampwidth() == 2
    sample_rate = wav.getframerate()
    pcm = np.frombuffer(wav.readframes(wav.getnframes()), dtype="<i2")
samples = pcm.astype(np.float32) / 32768
stream = recognizer.create_stream()
stream.accept_waveform(sample_rate, samples)
recognizer.decode_stream(stream)
print(stream.result.text)
```

Use a mono 16-bit PCM WAV. For Multilingual-v2, set `language="kk"` on the recognizer
or call `stream.set_option("language", "kk")` before decoding an individual
stream. For v4, leave the language at `"auto"`.

## Offline regression test

From the source checkout, configure CMake with `SHERPA_ONNX_ENABLE_PYTHON=ON`
and `SHERPA_ONNX_ENABLE_TESTS=ON` and build `_sherpa_onnx`. Install NumPy
in the Python environment used by the build. Download the pinned v4 INT8
graphs, E2E tokenizer, and a Sherpa Russian WAV with the Hugging Face CLI
(`python -m pip install 'huggingface_hub==1.3.5'`).

```sh
mkdir -p sherpa-onnx/python/sherpa_onnx/lib
cp build/lib/_sherpa_onnx*.so sherpa-onnx/python/sherpa_onnx/lib/
hf download ai-babai/gigaam-v4-onnx \
  rnnt/v4_rnnt_encoder.int8.onnx rnnt/v4_rnnt_advance.onnx rnnt/v4_rnnt_e2e_tokenizer.model \
  --revision d39e2a200f596315acef7003d246b65a73e9a327 --local-dir gigaam-v4
curl -fL -o gigaam-v4/test.wav \
  https://huggingface.co/spaces/k2-fsa/automatic-speech-recognition/resolve/2408de4949741cb03b444a8edbc3d38acd769927/test_wavs/russian/test.wav
export SHERPA_ONNX_GIGAAM_OFFLINE_PACKAGE="$PWD/gigaam-v4/rnnt"
export SHERPA_ONNX_GIGAAM_TEST_WAV="$PWD/gigaam-v4/test.wav"
ctest --test-dir build -R '^test_gigaam_offline_py$' --output-on-failure
```

The WAV SHA-256 is `d6e768803b3bc3afcb08326677f3bb872c9beeed29af40d3579bcc14e74484f8`.
The test checks a word from the transcript, compares single-stream and list
decoding, and rejects an advance graph passed as the encoder. CTest skips it
if the fixture variables are unset.
