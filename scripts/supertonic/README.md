# Supertonic TTS INT8 Quantization

Quantize [Supertonic](https://github.com/supertone-inc/supertonic) TTS ONNX models to INT8 for on-device deployment.

## Overview

- **Pipeline**: `gen_calib_configs` → `dump_inputs` → `convert`; voice style JSONs → single `voice.bin` via `generate_voices_bin.py`.
- **Quantization**: duration_predictor, text_encoder, vector_estimator → dynamic INT8; vocoder → static INT8 (calibration from dumped data).
- **Voice**: Runtime loads one `voice.bin` at init (all JSONs in `assets/voice_styles` merged); use `--sid` (0..N-1) to select speaker.

## Usage

```bash
./run.sh              # Run all stages (0–4)
./run.sh 4            # Only generate voice.bin
```

**Stages:** 0 = download models, 1 = gen calib configs, 2 = dump calib data, 3 = quantize, 4 = generate `voice.bin`.
