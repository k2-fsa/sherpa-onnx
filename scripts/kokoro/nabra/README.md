# Nabra-82M (Arabic, MSA) — sherpa-onnx packaging

This directory contains scripts to convert [Nabra-82M](https://huggingface.co/oddadmix/Nabra-82M-v0.1)
ONNX exports for use with sherpa-onnx's Kokoro runtime.

Nabra-82M is a Kokoro/StyleTTS2-architecture model fine-tuned for Modern
Standard Arabic. Its ONNX export shares the Kokoro v1.0 input signature
(`tokens`, `style[1,256]`, `speed`), so it runs on the existing Kokoro runtime
without code changes.

Pre-packaged models: https://huggingface.co/marwanelamami/nabra-82m-sherpa-onnx

## Usage

```bash
./run.sh /path/to/nabra_int4qat.onnx
```

Produces:
- `model.int4.onnx` — metadata-stamped + iSTFT notch FIR baked in
- `tokens.txt` — phoneme vocabulary from the model's `vocab.json`
- `voices.bin` — style vectors `[510, 1, 256]`

## Test

```python
import sherpa_onnx

tts = sherpa_onnx.OfflineTts(
    sherpa_onnx.OfflineTtsConfig(
        model=sherpa_onnx.OfflineTtsModelConfig(
            kokoro=sherpa_onnx.OfflineTtsKokoroModelConfig(
                model="model.int4.onnx",
                voices="voices.bin",
                tokens="tokens.txt",
                data_dir="espeak-ng-data",
            ),
        ),
    )
)
audio = tts.generate("السلام عليكم", sid=0, speed=1.0)
```
