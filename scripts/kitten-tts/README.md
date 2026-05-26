# Introduction

See also https://github.com/KittenML/KittenTTS

## KittenTTS v0.8

Use `v0_8/run.sh` to prepare a sherpa-onnx compatible model directory from
the upstream Hugging Face assets:

```bash
cd scripts/kitten-tts/v0_8
python3 -m pip install numpy onnx
./run.sh KittenML/kitten-tts-nano-0.8-fp32
./run.sh KittenML/kitten-tts-nano-0.8-int8
./run.sh KittenML/kitten-tts-micro-0.8
./run.sh KittenML/kitten-tts-mini-0.8
```

The generated `voices.bin` preserves all v0.8 reference rows. The ONNX metadata
marks v0.8 models with `version=8`, `end_id=10`, `add_pad_after_end=1`, and
`max_token_len=400` so the runtime selects the same style row as upstream.
Use package directories like `kitten-mini-en-v0_8`,
`kitten-micro-en-v0_8`, `kitten-nano-en-v0_8-fp32`, or
`kitten-nano-en-v0_8-int8`.
