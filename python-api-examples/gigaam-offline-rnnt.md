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
