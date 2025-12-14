# Introduction

Model files are from
https://modelscope.cn/models/dengcunqin/matcha_tts_zh_en_20251010/summary

Note that you have to use
vocos-16khz-univ.onnx

You can download it from
 https://modelscope.cn/models/dengcunqin/matcha_tts_zh_en_20251010/resolve/master/vocos-16khz-univ.onnx
or
 https://github.com/k2-fsa/sherpa-onnx/releases/download/vocoder-models/vocos-16khz-univ.onnx

```
{'am': './model-steps-3.onnx', 'vocoder': './vocos-16khz-univ.onnx', 'tokens': './tokens.txt', 'lexicon': './lexicon.txt', 'text': '中英文合成测试. It supports both English 和中文合成', 'out_wav': 'generated.wav'}

{'use_eos_bos': '1', 'modelscope_url': 'https://modelscope.cn/models/dengcunqin/matcha_tts_zh_en_20251010', 'sample_rate': '16000', 'language': 'chinese English', 'model_type': 'matcha-tts', 'n_speakers': '1', 'model_author': 'dengcunqin', 'version': '1', 'pad_id': '0', 'voice': 'zh en-us', 'demo_url': 'https://www.tulingyun.com/tts.html', 'num_ode_steps': '3'}

NodeArg(name='x', type='tensor(int64)', shape=['N', 'L'])
NodeArg(name='x_length', type='tensor(int64)', shape=['N'])
NodeArg(name='noise_scale', type='tensor(float)', shape=[1])
NodeArg(name='length_scale', type='tensor(float)', shape=[1])
-----
NodeArg(name='mel', type='tensor(float)', shape=['N', 80, 'L'])

vocos {'modelscope_url': 'https://modelscope.cn/models/dengcunqin/matcha_tts_zh_en_20251010', 'use_eos_bos': '1', 'n_speakers': '1', 'sample_rate': '16000', 'pad_id': '0', 'language': 'chinese English', 'model_type': 'matcha-tts vocos', 'voice': 'zh en-us', 'version': '1', 'demo_url': 'https://www.tulingyun.com/tts.html', 'model_author': 'dengcunqin'}

----------vocos----------
NodeArg(name='mels', type='tensor(float)', shape=['batch_size', 80, 'time'])
-----
NodeArg(name='mag', type='tensor(float)', shape=['batch_size', 'Clipmag_dim_1', 'time'])
NodeArg(name='x', type='tensor(float)', shape=['batch_size', 'Cosx_dim_1', 'time'])
NodeArg(name='y', type='tensor(float)', shape=['batch_size', 'Cosx_dim_1', 'time'])
```
