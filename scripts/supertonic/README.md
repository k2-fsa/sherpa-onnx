# Supertonic TTS INT8 Quantization

Quantize [Supertonic](https://github.com/supertone-inc/supertonic) TTS ONNX models (duration_predictor, text_encoder, vector_estimator, vocoder) to INT8 for on-device deployment.

## Overview

- **Three scripts**: `gen_calib_configs` → `dump_inputs` → `convert`
- Static INT8 needs real activations, so we dump them first; dynamic INT8 does not
- **Quantization**: duration_predictor & text_encoder → dynamic INT8; vector_estimator → dynamic INT8; vocoder → static INT8 on Conv (last few convs stay FP32 for quality)
- Dynamic INT8: scales per inference, no calibration. Static INT8: fixed scales from calibration data.
- Calibration uses percentile-based pad/crop for variable-length sequences.
- Vocoder calibration derived from vector_estimator output to avoid extra inference.
- W8-DQ (weight-only int8 + dequant at runtime) on some Conv layers for extra compression.

## Usage

```bash
./run.sh              # Run all stages (0–3)
```

Stages: 0 = download models, 1 = gen calib configs, 2 = dump calib data, 3 = quantize.
