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
            acoustic_model="matcha-icefall-zh-en/model-steps-3.onnx",
            vocoder="vocos-16khz-univ.onnx",
            lexicon="matcha-icefall-zh-en/lexicon.txt",
            tokens="matcha-icefall-zh-en/tokens.txt",
            data_dir="matcha-icefall-zh-en/espeak-ng-data",
        ),
        num_threads=2,
    ),
    max_num_sentences=1,
    rule_fsts="./matcha-icefall-zh-en/phone-zh.fst,./matcha-icefall-zh-en/date-zh.fst,./matcha-icefall-zh-en/number-zh.fst",
)

if not config.validate():
    raise ValueError("Please check your config")

tts = sherpa_onnx.OfflineTts(config)
text = "我最近在学习machine learning，希望能够在未来的artificial intelligence领域有所建树。在这次vacation中，我们计划去Paris欣赏埃菲尔铁塔和卢浮宫的美景。某某银行的副行长和一些行政领导表示，他们去过长江和长白山; 经济不断增长。开始数字测试。2025年12月4号，拨打110或者189202512043。123456块钱。在这个快速发展的时代，人工智能技术正在改变我们的生活方式。语音合成作为人工智能的重要应用之一，让机器能够用自然流畅的语音与人类进行交流。"


audio = tts.generate(text, sid=0, speed=1.0)

sf.write(
    "./hf/matcha/icefall-zh-en/mp3/0.mp3",
    audio.samples,
    samplerate=audio.sample_rate,
)
