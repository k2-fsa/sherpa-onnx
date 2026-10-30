# Introduction

- [./convert_tokenizer.py](./convert_tokenizer.py) It produces `./token_scores.json`
  and `./vocab.json` from [./tokenizer.model](https://huggingface.co/KevinAHM/pocket-tts-onnx/resolve/main/tokenizer.model)

- [./test_tokenizer.py](./test_tokenizer.py) is used to test the exported `./token_scores.json`
  and `./vocab.json`

- [./add_bos_before_voice.py](./add_bos_before_voice.py) makes the multilingual bundles of
  [KevinAHM/pocket-tts-onnx](https://huggingface.co/KevinAHM/pocket-tts-onnx/tree/main/onnx)
  (`german`, `french_24l`, `spanish`, `italian`, `portuguese`, ...) work with sherpa-onnx.
  These bundles set `"insert_bos_before_voice": true` in `bundle.json`: the embedding in
  `bos_before_voice.npy` has to go in front of the voice embedding, otherwise the model does not
  stop and produces noise. The script appends it inside the mimi encoder, so no C++ change is
  needed. Use its output as `--pocket-encoder`; the other files of the bundle are used as they are
  (`flow_lm_flow_int8.onnx` as `--pocket-lm-flow`, `flow_lm_main_int8.onnx` as `--pocket-lm-main`,
  `mimi_decoder_int8.onnx` as `--pocket-decoder`, `text_conditioner_int8.onnx` as
  `--pocket-text-conditioner`, and `vocab.json`/`token_scores.json` from `./convert_tokenizer.py`
  run on the bundle's own `tokenizer.model`).

In C++, we don't need to use the [sentencepiece](https://github.com/google/sentencepiece) or protobuf for the tokenizer.
