# Supertonic TTS INT8 Quantization

Quantize [Supertonic](https://github.com/supertone-inc/supertonic) TTS ONNX models to INT8 for on-device deployment.

## Overview

- **Pipeline**: `gen_calib_configs` → `dump_inputs` → `convert`; stage 4 generates **.bin** assets when JSONs exist: `generate_voices_bin.py`, `generate_indexer_bin.py`. Runtime loads **tts.json** for TTS config.
- **Quantization**: duration_predictor, text_encoder, vector_estimator → dynamic INT8; vocoder → static INT8 (calibration from dumped data).
- **Voice**: Runtime loads one **`voice.bin`**. Generate with `python3 generate_voices_bin.py [input_dir] [output_bin]`. Pass `--supertonic-voice-style=/path/to/voice.bin`. Use `--sid` 0..N-1 to select speaker.
- **Unicode indexer**: Runtime uses **`unicode_indexer.bin`**. Generate with `python3 generate_indexer_bin.py [json_path] [bin_path]`. Pass `--supertonic-unicode-indexer=/path/to/unicode_indexer.bin`.
- **TTS config**: Runtime loads **`tts.json`**. Pass `--supertonic-tts-json=/path/to/tts.json`.

## Usage

```bash
./run.sh              # Run all stages (0–4)
./run.sh 4            # Only generate voice.bin, unicode_indexer.bin
```

**Stages:** 0 = download models, 1 = gen calib configs, 2 = dump calib data, 3 = quantize, 4 = generate `voice.bin`, `unicode_indexer.bin`. 
