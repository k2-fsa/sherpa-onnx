# Parakeet v3 frontend validation

The native frontend targets **current NVIDIA inference preprocessing**, rather
than every historical NeMo release. The checkpoint records `2.3.0rc5`; NeMo
2.3.0 uses reflected STFT padding and one more valid frame. NVIDIA subsequently
changed these choices to constant padding and `floor(samples / hop)` in
[the padding/batch invariance correction](https://github.com/NVIDIA/NeMo/commit/0fd4de534b7b6ad59271850904d0759deac3fd6a).
The comparison pins both current NVIDIA source and NeMo 2.3.0, and reports
the difference explicitly. It does not claim parity with the original training
runtime.

Build the model-free tests and feature probe:

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=ON \
  -DSHERPA_ONNX_ENABLE_TESTS=ON -DSHERPA_ONNX_ENABLE_BINARY=ON \
  -DSHERPA_ONNX_ENABLE_PORTAUDIO=OFF -DSHERPA_ONNX_ENABLE_WEBSOCKET=OFF \
  -DSHERPA_ONNX_ENABLE_TTS=OFF -DSHERPA_ONNX_ENABLE_SPEAKER_DIARIZATION=OFF \
  -DSHERPA_ONNX_BUILD_C_API_EXAMPLES=OFF
cmake --build build --target parakeet-feature-dump \
  offline-stream-parakeet-test sherpa-onnx-offline -j 8
ctest --test-dir build -R '^offline-stream-parakeet-test$' --output-on-failure
```

Set up an isolated Python environment (the comparison was run with Python 3.12):

```sh
python3 -m venv build/parakeet-reference-env
build/parakeet-reference-env/bin/pip install \
  torch==2.10.0+cpu --index-url https://download.pytorch.org/whl/cpu
build/parakeet-reference-env/bin/pip install \
  numpy==2.2.6 librosa==0.11.0 PyYAML==6.0.2 soundfile==0.14.0 \
  onnxruntime==1.30.0
```

The model-free comparison needs only small source/configuration/audio downloads:

```sh
build/parakeet-reference-env/bin/python scripts/parakeet/compare-reference.py \
  --probe build/bin/parakeet-feature-dump \
  --output-dir build/parakeet-feature-comparison
```

For a full-model comparison, add the following arguments. This downloads the
public FP32 ONNX export, including its approximately 2.4 GB external weights:

```sh
build/parakeet-reference-env/bin/python scripts/parakeet/compare-reference.py \
  --probe build/bin/parakeet-feature-dump \
  --output-dir build/parakeet-model-comparison \
  --model-dir build/parakeet-public-model --download-model \
  --native-offline build/bin/sherpa-onnx-offline
```

An existing model directory can be supplied without `--download-model`. Its
files must match the pinned public export; the script validates their hashes.
Use a fresh output directory for each completed run.

`results.json` records source/model revisions, hashes, checkpoint configuration,
package versions, feature errors and frame counts, and optional transcripts and
token IDs. The fixtures are the export's four public English/German/Spanish/French
WAVs, an English copy attenuated by 60 dB, four deterministic broadband boundary
inputs, and digital silence. Non-16 kHz public WAVs use sherpa's native resampler
before **all** frontend comparisons.

The reference executes NVIDIA's original `FilterbankFeatures`, `normalize_batch`
and `splice_frames` AST nodes from the checksum-verified source. No feature math
is rewritten. This narrow execution avoids importing the unrelated NeMo training,
model and augmentation stack. A type-annotation placeholder is provided for the
unused packed-inference path. The comparison uses the original dense `forward`
in evaluation mode, where NeMo disables dither, with the checkpoint's settings.
It is a source-level extractor comparison, not a test of the complete installed
NeMo application.

For decoding, identical FP32 weights and the same greedy TDT loop process the
legacy native, corrected native and current NVIDIA features. Historical NeMo
features are additionally decoded on the English sample. The real native CLI
is checked separately on all five speech fixtures. The feature probe's `auto`
mode also constructs the recognizer from public model metadata and verifies
that its selected features exactly equal the corrected frontend's features.
Native and Python ONNX Runtime versions can differ; their versions are recorded
in the evidence. These checks establish parity and exercise the integrated
runtime, but do not establish a general word-error-rate improvement.

The script fails on frame-count differences, non-finite corrected features,
current-reference RMS error at or above `3e-4` or maximum absolute error at or
above `3e-3`,
different corrected/reference token sequences, incorrect model-metadata
selection, or a native CLI transcript differing from the current-reference
transcript. Historical differences are reported, not treated as current-reference
failures. The probe also decodes 0/1/159/160/319-sample inputs in short-only and
mixed speech/short batches: short results must be empty, and speech must retain
its individually decoded text and tokens. Unit tests cover waveform chunk
boundaries, resampling and shorter-than-two-frame feature extraction.

The native implementation accumulates normalization statistics in float64.
NVIDIA's float32 mean can round noticeably relative to a nearly constant bin's
small standard deviation, especially on quiet or two-frame inputs. The script
reports a separate diagnostic that keeps NVIDIA's original STFT/mel/log code
and computes the mean and sample variance in float64 with NumPy. This diagnostic
must meet the stricter `1e-4` RMS and `1e-3` maximum-error bounds. It also records
NVIDIA float32 versus this diagnostic, so reviewers can distinguish rounding
from frontend differences. This additional normalization calculation is our
diagnostic, not unmodified NVIDIA code. The primary comparison and all reference
decoding still use NVIDIA's unchanged float32 `forward`.

The direct-DFT fixture remains an additional check independent of both PyTorch
and kaldi-native-fbank:

```sh
python3 scripts/parakeet/generate-reference.py > /tmp/parakeet-reference.inc
diff -u sherpa-onnx/csrc/offline-stream-parakeet-reference.inc /tmp/parakeet-reference.inc
```

Observed on Linux x86_64 CPU with GCC 15, native ONNX Runtime 1.28.2,
Python ONNX Runtime 1.30.0, and the versions/revisions above:

| Input | Valid frames | RMS error vs NVIDIA float32 | Maximum absolute error |
| --- | ---: | ---: | ---: |
| en | 384 | 2e-06 | 3.54e-05 |
| de | 275 | 2.03e-06 | 0.000134 |
| es | 532 | 1.57e-06 | 4.36e-05 |
| fr | 496 | 2.69e-06 | 0.000168 |
| en-quiet | 384 | 0.000178 | 0.00198 |
| broadband-320 | 2 | 1.5e-05 | 0.000193 |
| broadband-321 | 2 | 0.000141 | 0.00175 |
| broadband-1281 | 8 | 7.82e-06 | 5.16e-05 |
| broadband-16001 | 100 | 8.13e-06 | 0.000113 |
| silence | 100 | 0 | 0 |

All ten comparisons pass, including the stricter double-normalization diagnostic.
The five speech fixtures produce identical corrected/current-reference token IDs;
the native CLI produces the same text. Legacy/current common-prefix RMS errors
on the four original public WAVs range from 0.345 to 0.411, with one additional
legacy frame. The legacy frontend produces empty text on the 60 dB attenuated
English sample; the corrected and reference frontends both produce:

> Ask not what your country can do for you. Ask what you can do for your country.

Three original public WAVs retain the legacy transcript. Spanish changes `King`
to `quién`, matching the reference, but the sentence still contains an error.
These are a small parity/reproduction set, not an accuracy benchmark. No GPU,
ARM64, alternate exports, or long-form segmentation validation is claimed.
