# Intel NPU support with OpenVINO

Sherpa-onnx can run ONNX models on Intel NPUs through ONNX Runtime's OpenVINO
Execution Provider. This requires ONNX Runtime 1.17 or newer built with the
OpenVINO provider; the standard ONNX Runtime package bundled by sherpa-onnx
does not include it.

A standalone OpenVINO installation is not sufficient by itself: the ONNX
Runtime library must also be built with the OpenVINO Execution Provider, and
that build must be compatible with the installed OpenVINO runtime.

## Build

Install an OpenVINO-enabled ONNX Runtime and initialize the OpenVINO runtime
environment. Then point sherpa-onnx at its C/C++ headers and libraries:

```bash
source /path/to/openvino/setupvars.sh
export SHERPA_ONNXRUNTIME_INCLUDE_DIR=/path/to/onnxruntime/include
export SHERPA_ONNXRUNTIME_LIB_DIR=/path/to/onnxruntime/lib

cmake -S . -B build-openvino \
  -DBUILD_SHARED_LIBS=ON \
  -DSHERPA_ONNX_USE_PRE_INSTALLED_ONNXRUNTIME_IF_AVAILABLE=ON
cmake --build build-openvino --parallel
```

`OpenVINOExecutionProvider` should appear in the available-provider list when
the OpenVINO-enabled ONNX Runtime and its dependencies are discoverable.

## Use the Intel NPU for wake-word detection

Set the existing sherpa-onnx provider option to `openvino`. The default
OpenVINO device is `NPU`:

```bash
build-openvino/bin/sherpa-onnx-keyword-spotter \
  --provider=openvino \
  --tokens=/path/to/tokens.txt \
  --encoder=/path/to/encoder.onnx \
  --decoder=/path/to/decoder.onnx \
  --joiner=/path/to/joiner.onnx \
  --keywords-file=/path/to/keywords.txt \
  /path/to/audio.wav
```

For transducer keyword-spotting models, sherpa-onnx runs the compute-heavy
encoder on OpenVINO and keeps the small decoder and joiner on CPU. This avoids
unnecessary NPU dispatch overhead and preserves decoder accuracy. The Silero
VAD model tested for this integration ran entirely on OpenVINO. Other VAD
models may be partitioned by ONNX Runtime, with unsupported nodes assigned to
the CPU provider.

For a microphone, use `sherpa-onnx-keyword-spotter-microphone` (PortAudio) or
`sherpa-onnx-keyword-spotter-alsa` with the same `--provider` value.

## Use the Intel NPU for VAD

VAD uses the `--vad-provider` option:

```bash
build-openvino/bin/sherpa-onnx-vad \
  --vad-provider=openvino \
  --silero-vad-model=/path/to/silero_vad.onnx \
  /path/to/input.wav \
  /path/to/speech-only.wav
```

The microphone VAD binaries accept the same `--vad-provider` value.

The same provider string works through the C++, C, Python, Java, Kotlin, Dart,
Go, Rust, Swift, C#, and Node.js APIs. For example, Python model configs accept
`provider="openvino"`.

If OpenVINO is unavailable or cannot initialize, sherpa-onnx logs the reason
and falls back to the ONNX Runtime CPU provider.

## Provider options

OpenVINO options use sherpa-onnx's existing `provider:config-file` syntax. The
file contains one `key=value` entry per line:

```ini
# openvino-npu.config
device_type=NPU
enable_qdq_optimizer=True
```

Pass it by appending the path to the provider name. Use `--provider` for
keyword spotting and `--vad-provider` for VAD:

```bash
--provider=openvino:openvino-npu.config
--vad-provider=openvino:openvino-npu.config
```

Any OpenVINO V2 provider option can be used. `device_type` may also be `CPU`,
`GPU`, or `AUTO`. Multi-device modes require at least two devices; for example,
use `HETERO:GPU,CPU` or `MULTI:GPU,CPU`. Bare `HETERO` and `MULTI` values are
incomplete device selections. `AUTO:GPU,NPU,CPU` can be used to give automatic
selection an explicit device priority. If `GraphOptimizationLevel` is omitted,
sherpa-onnx disables ONNX Runtime graph optimizations as recommended by the
OpenVINO Execution Provider. Set it explicitly in the config file to override
that behavior.

OpenVINO's NPU support depends on the model's operators and shapes. Dynamic
speech-model inputs may need `disable_dynamic_shapes` and `reshape_input`
bounds tailored to the selected model; unsupported work can execute on the CPU
fallback provider.

See the [ONNX Runtime OpenVINO Execution Provider documentation][ort-openvino]
for installation packages, supported devices, options, and compatibility.

[ort-openvino]: https://onnxruntime.ai/docs/execution-providers/OpenVINO-ExecutionProvider.html
