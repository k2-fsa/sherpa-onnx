# Introduction

- [./convert_tokenizer.py](./convert_tokenizer.py) It produces `./token_scores.json`
  and `./vocab.json` from [./tokenizer.model](https://huggingface.co/KevinAHM/pocket-tts-onnx/resolve/main/tokenizer.model)

- [./test_tokenizer.py](./test_tokenizer.py) is used to test the exported `./token_scores.json`
  and `./vocab.json`

In C++, we don't need to use the [sentencepiece](https://github.com/google/sentencepiece) or prootobuf for the tokenizer.
