# Parakeet v3 and Ultra frontend validation

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
ARM64 or long-form segmentation validation is claimed. The separately tested
Ultra export is described below.


## Automatic Ultra selection

The same frontend is selected for NVIDIA Parakeet TDT 0.6B v3 and Moondream
Parakeet Ultra when the encoder declares 128 features, normalization is
`per_feature`, and the configured sample rate is 16 kHz. Selection uses the
existing encoder `url` metadata. No new metadata, re-export, or model download
is required for an already installed supported export after updating the runtime.

For Ultra the recognized URL values are:

- `https://huggingface.co/moondream/parakeet-ultra`
- `https://huggingface.co/moondream/parakeet-ultra (parakeet-tdt architecture)`
- `https://huggingface.co/moondream/parakeet-ultra (parakeet-tdt-0.6b-v3 architecture)`
- `https://huggingface.co/moondream/parakeet-ultra (retrained nvidia/parakeet-tdt-0.6b-v3)`

These cover the plain identity and the architecture annotations used by the
mldecode, tpmaerk and TypeWhisper exporters. The plain identity also selects
the TDT decoder without requiring `tdt` inside the URL. Other metadata identities,
including v2 and similarly named models, retain the previous frontend behavior.
This is not a blanket rule for every Parakeet model or arbitrary custom export.

## Ultra reproduction

`compare-ultra.py` pins the public
[tpmaerk/parakeet-ultra-sherpa-onnx-int8](https://huggingface.co/tpmaerk/parakeet-ultra-sherpa-onnx-int8/tree/b86784a3a5356cbd126f710b61cd90c5fe5898f1)
export and verifies all four file hashes. It separately retrieves the config
from the converted
[Ultra NeMo checkpoint](https://huggingface.co/Olicorne/parakeet-tdt-0.6b-v3-ultra-onnx/tree/a1fe742758beb8ddc918bff00e1e319fe7e4d3e3).
That config is byte-identical to the v3 config used above. This establishes
which configuration was tested, not independent evidence of Ultra's training
preprocessing or Moondream's Photon implementation.

```sh
build/parakeet-reference-env/bin/pip install onnx==1.20.1
build/parakeet-reference-env/bin/python scripts/parakeet/compare-ultra.py \
  --probe build/bin/parakeet-feature-dump \
  --native-offline build/bin/sherpa-onnx-offline \
  --model-dir build/ultra-public-model \
  --output-dir build/ultra-comparison
```

This downloads about 670 MB of model files. Speech decoding and automatic
selection use the original export unchanged. Separate temporary copies vary
only metadata to check the recognized URL spellings and unrelated-model
fallback. Use a fresh output directory for each run; verified downloads can
be reused. `results.json` records original metadata, revisions, hashes,
transcripts, token IDs, native CLI output and selection checks. The nested
`features/results.json` records the ten feature comparisons and environment.

On Linux x86_64 CPU, the legacy frontend with these int8 Ultra weights produces
an empty transcript for the public English sample multiplied by `0.001`.
The corrected native frontend and pinned NVIDIA extractor produce:

> Ask not what your country can do for you. Ask what you can do for your country.

All five speech cases have identical corrected/NVIDIA token IDs. Native CLI
text matches the reference decode with the untouched export. Ten feature
checks, short-only and mixed short/speech batches, and eight model-selection
cases pass, including TDT decoding with the plain Ultra URL.

This is a reproduction and parity test, not an overall WER benchmark. On the
normal-volume Spanish sample, the corrected frontend omits `tú`, which the
legacy frontend includes; both corrected and NVIDIA-reference decoding make
that omission. French punctuation also changes. Matching the reference does
not imply every utterance improves. Full decoding validation covers the pinned
tpmaerk int8 export; the other recognized URL spellings are exercised with
metadata-only copies of that model, not independent accuracy evaluations of
every publisher's weights. No GPU, ARM64, Photon or long-form validation is
claimed for Ultra.
