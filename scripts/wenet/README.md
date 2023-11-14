# Introduction

This folder contains script for exporting models
from [wenet](https://github.com/wenet-e2e/wenet)
to onnx. You can use the exported models in sherpa-onnx.

Note that both **streaming** and **non-streaming** models are supported.

We only use the CTC branch. Rescore with the attention decoder
is not supported, though decoding with H, HL, and HLG is supported.
