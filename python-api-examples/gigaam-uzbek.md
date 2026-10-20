# Uzbek speech recognition with GigaAM

[Alex13/sherpa-onnx-gigaam-uzbek-600m](https://huggingface.co/Alex13/sherpa-onnx-gigaam-uzbek-600m)
is an INT8 ONNX export of the `large_full_600m` checkpoint from
[rustam1221/uzbek-asr-gigaam](https://huggingface.co/rustam1221/uzbek-asr-gigaam),
an Uzbek fine-tune of
[ai-sage/GigaAM-Multilingual](https://huggingface.co/ai-sage/GigaAM-Multilingual).
Training is credited to rustam1221; the sherpa-onnx export is by Alex13.
The export's model card declares an MIT license.

This is an offline (non-streaming) speech recognition model. It uses the
existing NeMo CTC backend, with GigaAM preprocessing selected by the ONNX
metadata `is_giga_am=1`. No language flag or new runtime backend is required.
The model card describes lowercase output with punctuation.

## Download

From the repository root, download the model and its matching vocabulary.
The model download is approximately 592 MB. These URLs pin both files to the
same Hugging Face revision.

```bash
mkdir -p sherpa-onnx-gigaam-uzbek-600m
base=https://huggingface.co/Alex13/sherpa-onnx-gigaam-uzbek-600m/resolve/b53434a0b06418831dbc1cec73d6b851054f593d
curl -fL "$base/model.int8.onnx" \
  -o sherpa-onnx-gigaam-uzbek-600m/model.int8.onnx
curl -fL "$base/tokens.txt" \
  -o sherpa-onnx-gigaam-uzbek-600m/tokens.txt
```

## Python example

Install the example dependencies in your Python environment:

```bash
python3 -m pip install sherpa-onnx soundfile
```

Use your own Uzbek speech recording; the model repository does not include
test audio.

```bash
python3 python-api-examples/offline-gigaam-uzbek-ctc-decode-files.py \
  --model-dir ./sherpa-onnx-gigaam-uzbek-600m \
  recording.wav
```

You can supply multiple audio filenames. The example processes files one at a
time on the CPU and prints each filename and transcript. It reads the first
channel of multichannel recordings. The model frontend operates at 16 kHz;
sherpa-onnx resamples input when necessary. Use short utterances or split long
recordings into speech segments with VAD before decoding.

## Command-line example

With a built `sherpa-onnx-offline` executable:

```bash
./build/bin/sherpa-onnx-offline \
  --nemo-ctc-model=./sherpa-onnx-gigaam-uzbek-600m/model.int8.onnx \
  --tokens=./sherpa-onnx-gigaam-uzbek-600m/tokens.txt \
  --num-threads=2 \
  recording.wav
```

The model and vocabulary can also be used through the existing NeMo CTC
configuration in other language bindings.
