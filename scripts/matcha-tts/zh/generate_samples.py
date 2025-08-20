#!/usr/bin/env python3
# Copyright    2025  Xiaomi Corp.        (authors: Fangjun Kuang)
"""
Generate samples for
https://k2-fsa.github.io/sherpa/onnx/tts/all/
"""


import sherpa_onnx
import soundfile as sf

config = sherpa_onnx.OfflineTtsConfig(
    model=sherpa_onnx.OfflineTtsModelConfig(
        matcha=sherpa_onnx.OfflineTtsMatchaModelConfig(
            acoustic_model="matcha-icefall-zh-baker/model-steps-3.onnx",
            vocoder="vocos-22khz-univ.onnx",
            lexicon="matcha-icefall-zh-baker/lexicon.txt",
            tokens="matcha-icefall-zh-baker/tokens.txt",
            dict_dir="matcha-icefall-zh-baker/dict",
        ),
        num_threads=2,
    ),
    max_num_sentences=1,
    rule_fsts="./matcha-icefall-zh-baker/phone.fst,./matcha-icefall-zh-baker/date.fst,./matcha-icefall-zh-baker/number.fst",
)

if not config.validate():
    raise ValueError("Please check your config")

tts = sherpa_onnx.OfflineTts(config)
text = "某某银行的副行长和一些行政领导表示，他们去过长江和长白山; 经济不断增长。2024年12月31号，拨打110或者18920240511。123456块钱。当夜幕降临，星光点点，伴随着微风拂面，我在静谧中感受着时光的流转，思念如涟漪荡漾，梦境如画卷展开，我与自然融为一体，沉静在这片宁静的美丽之中，感受着生命的奇迹与温柔."


audio = tts.generate(text, sid=0, speed=1.0)

sf.write(
    "./hf/matcha/icefall-zh/mp3/0.mp3",
    audio.samples,
    samplerate=audio.sample_rate,
)
