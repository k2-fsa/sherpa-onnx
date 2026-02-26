# Supertonic TTS INT8 Quantization

Quantize [Supertonic](https://github.com/supertone-inc/supertonic) TTS ONNX models to INT8 for on-device deployment.

## Overview

- **Pipeline**: `gen_calib_configs` → `dump_inputs` → `convert`; stage 4 generates all **.bin** assets from JSONs when present. Scripts take explicit input/output where applicable: `generate_voices_bin.py [input_dir] [output_bin]`, `generate_indexer_bin.py [json_path] [bin_path]`, `generate_tts_bin.py [json_path] [bin_path]`.
- **Quantization**: duration_predictor, text_encoder, vector_estimator → dynamic INT8; vocoder → static INT8 (calibration from dumped data).
- **Voice**: Runtime loads one **`voice.bin`**. Generate with `python3 generate_voices_bin.py [input_dir] [output_bin]` (default: `./assets/voice_styles` → `./assets/voice_styles/voice.bin`). Pass `--supertonic-voice-style=/path/to/voice.bin`. Use `--sid` 0..N-1 to select speaker.
- **Unicode indexer**: Runtime uses **`unicode_indexer.bin`** only. Generate with `python3 generate_indexer_bin.py [json_path] [bin_path]`. Pass `--supertonic-unicode-indexer=/path/to/unicode_indexer.bin`.
- **TTS config**: Runtime uses a single TTS config file (generated as `tts.bin` by `generate_tts_bin.py`). Generate with `python3 generate_tts_bin.py [json_path] [bin_path]`. Pass `--supertonic-tts-config=/path/to/tts.bin`.

## Usage

```bash
./run.sh              # Run all stages (0–4)
./run.sh 4            # Only generate voice.bin, unicode_indexer.bin, tts.bin (when JSONs exist)
```

**Stages:** 0 = download models, 1 = gen calib configs, 2 = dump calib data, 3 = quantize, 4 = generate `voice.bin`, `unicode_indexer.bin`, `tts.bin`.
