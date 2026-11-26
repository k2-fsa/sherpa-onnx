# sherpa-onnx/python/tests/test_online_recognizer.py
#
# Copyright (c)  2023  Xiaomi Corporation
#
# To run this single test, use
#
#  ctest --verbose -R  test_online_recognizer_py

import re
import unittest
import wave
from pathlib import Path
from typing import Tuple

import numpy as np
import sherpa_onnx

d = "/tmp/icefall-models"
# Please refer to
# https://k2-fsa.github.io/sherpa/onnx/pretrained_models/online-transducer/index.html
# to download pre-trained models for testing


def read_wave(wave_filename: str) -> Tuple[np.ndarray, int]:
    """
    Args:
      wave_filename:
        Path to a wave file. It should be single channel and each sample should
        be 16-bit. Its sample rate does not need to be 16kHz.
    Returns:
      Return a tuple containing:
       - A 1-D array of dtype np.float32 containing the samples, which are
       normalized to the range [-1, 1].
       - sample rate of the wave file
    """

    with wave.open(wave_filename) as f:
        assert f.getnchannels() == 1, f.getnchannels()
        assert f.getsampwidth() == 2, f.getsampwidth()  # it is in bytes
        num_samples = f.getnframes()
        samples = f.readframes(num_samples)
        samples_int16 = np.frombuffer(samples, dtype=np.int16)
        samples_float32 = samples_int16.astype(np.float32)

        samples_float32 = samples_float32 / 32768
        return samples_float32, f.getframerate()


class TestOnlineRecognizer(unittest.TestCase):
    def test_transducer_single_file(self):
        for use_int8 in [True, False]:
            if use_int8:
                encoder = f"{d}/sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20/encoder-epoch-99-avg-1.int8.onnx"
                decoder = f"{d}/sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20/decoder-epoch-99-avg-1.onnx"
                joiner = f"{d}/sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20/joiner-epoch-99-avg-1.int8.onnx"
            else:
                encoder = f"{d}/sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20/encoder-epoch-99-avg-1.onnx"
                decoder = f"{d}/sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20/decoder-epoch-99-avg-1.onnx"
                joiner = f"{d}/sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20/joiner-epoch-99-avg-1.onnx"

            tokens = f"{d}/sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20/tokens.txt"
            wave0 = f"{d}/sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20/test_wavs/0.wav"

            if not Path(encoder).is_file():
                print("skipping test_transducer_single_file()")
                return

            for decoding_method in ["greedy_search", "modified_beam_search"]:
                recognizer = sherpa_onnx.OnlineRecognizer.from_transducer(
                    encoder=encoder,
                    decoder=decoder,
                    joiner=joiner,
                    tokens=tokens,
                    num_threads=1,
                    decoding_method=decoding_method,
                    provider="cpu",
                )
                s = recognizer.create_stream()
                samples, sample_rate = read_wave(wave0)
                s.accept_waveform(sample_rate, samples)

                tail_paddings = np.zeros(int(0.2 * sample_rate), dtype=np.float32)
                s.accept_waveform(sample_rate, tail_paddings)

                s.input_finished()
                while recognizer.is_ready(s):
                    recognizer.decode_stream(s)
                print(recognizer.get_result(s))

    def test_transducer_multiple_files(self):
        for use_int8 in [True, False]:
            if use_int8:
                encoder = f"{d}/sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20/encoder-epoch-99-avg-1.int8.onnx"
                decoder = f"{d}/sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20/decoder-epoch-99-avg-1.onnx"
                joiner = f"{d}/sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20/joiner-epoch-99-avg-1.int8.onnx"
            else:
                encoder = f"{d}/sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20/encoder-epoch-99-avg-1.onnx"
                decoder = f"{d}/sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20/decoder-epoch-99-avg-1.onnx"
                joiner = f"{d}/sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20/joiner-epoch-99-avg-1.onnx"

            tokens = f"{d}/sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20/tokens.txt"
            wave0 = f"{d}/sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20/test_wavs/0.wav"
            wave1 = f"{d}/sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20/test_wavs/1.wav"
            wave2 = f"{d}/sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20/test_wavs/2.wav"
            wave3 = f"{d}/sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20/test_wavs/3.wav"
            wave4 = f"{d}/sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20/test_wavs/8k.wav"

            if not Path(encoder).is_file():
                print("skipping test_transducer_multiple_files()")
                return

            for decoding_method in ["greedy_search", "modified_beam_search"]:
                recognizer = sherpa_onnx.OnlineRecognizer.from_transducer(
                    encoder=encoder,
                    decoder=decoder,
                    joiner=joiner,
                    tokens=tokens,
                    num_threads=1,
                    decoding_method=decoding_method,
                    provider="cpu",
                )
                streams = []
                waves = [wave0, wave1, wave2, wave3, wave4]
                for wave in waves:
                    s = recognizer.create_stream()
                    samples, sample_rate = read_wave(wave)
                    s.accept_waveform(sample_rate, samples)

                    tail_paddings = np.zeros(int(0.2 * sample_rate), dtype=np.float32)
                    s.accept_waveform(sample_rate, tail_paddings)
                    s.input_finished()
                    streams.append(s)

                while True:
                    ready_list = []
                    for s in streams:
                        if recognizer.is_ready(s):
                            ready_list.append(s)
                    if len(ready_list) == 0:
                        break
                    recognizer.decode_streams(ready_list)
                results = [recognizer.get_result(s) for s in streams]
                for wave_filename, result in zip(waves, results):
                    print(f"{wave_filename}\n{result}")
                    print("-" * 10)

    def test_nemotron_streaming_english(self):
        m = "sherpa-onnx-nemotron-speech-streaming-en-0.6b-560ms-int8-2026-04-25"
        encoder = f"{d}/{m}/encoder.int8.onnx"
        decoder = f"{d}/{m}/decoder.int8.onnx"
        joiner = f"{d}/{m}/joiner.int8.onnx"
        tokens = f"{d}/{m}/tokens.txt"
        wave0 = f"{d}/{m}/test_wavs/0.wav"

        if not Path(encoder).is_file():
            print("skipping test_nemotron_streaming_english()")
            return

        recognizer = sherpa_onnx.OnlineRecognizer.from_transducer(
            encoder=encoder,
            decoder=decoder,
            joiner=joiner,
            tokens=tokens,
            num_threads=1,
            provider="cpu",
        )
        s = recognizer.create_stream()
        samples, sample_rate = read_wave(wave0)
        s.accept_waveform(sample_rate, samples)

        tail_paddings = np.zeros(int(0.2 * sample_rate), dtype=np.float32)
        s.accept_waveform(sample_rate, tail_paddings)

        s.input_finished()
        while recognizer.is_ready(s):
            recognizer.decode_stream(s)
        result = recognizer.get_result(s)
        print(result)
        self.assertTrue(len(result) > 0)

    def test_nemotron_multilingual_streaming(self):
        m = "sherpa-onnx-nemotron-3.5-asr-streaming-0.6b-560ms-int8-2026-06-11"
        encoder = f"{d}/{m}/encoder.int8.onnx"
        decoder = f"{d}/{m}/decoder.int8.onnx"
        joiner = f"{d}/{m}/joiner.int8.onnx"
        tokens = f"{d}/{m}/tokens.txt"
        ja_wave = f"{d}/{m}/test_wavs/ja.wav"
        en_wave = f"{d}/{m}/test_wavs/en.wav"

        if not Path(encoder).is_file():
            print("skipping test_nemotron_multilingual_streaming()")
            return

        recognizer = sherpa_onnx.OnlineRecognizer.from_transducer(
            encoder=encoder,
            decoder=decoder,
            joiner=joiner,
            tokens=tokens,
            num_threads=1,
            provider="cpu",
        )

        def decode(wave_filename: str, language: str = ""):
            s = recognizer.create_stream()
            if language:
                s.set_option("language", language)

            samples, sample_rate = read_wave(wave_filename)
            s.accept_waveform(sample_rate, samples)

            tail_paddings = np.zeros(int(0.2 * sample_rate), dtype=np.float32)
            s.accept_waveform(sample_rate, tail_paddings)

            s.input_finished()
            while recognizer.is_ready(s):
                recognizer.decode_stream(s)

            return recognizer.get_result(s)

        ja = decode(ja_wave, "ja")
        auto = decode(ja_wave, "auto")
        unset = decode(ja_wave)
        en = decode(en_wave, "en")

        print(f"ja: {ja}")
        print(f"auto: {auto}")
        print(f"unset: {unset}")
        print(f"en: {en}")

        self.assertTrue(len(ja) > 0)
        self.assertTrue(len(auto) > 0)
        self.assertTrue(len(unset) > 0)
        self.assertTrue(len(en) > 0)
        self.assertTrue(any(ord(c) > 127 for c in ja))
        for result in [ja, auto, unset, en]:
            self.assertIsNone(re.search(r"<[a-z]{2,3}(-[A-Z]{2})?>", result))

    def test_nemo_streaming_log_zero_guard(self):
        ctc = f"{d}/sherpa-onnx-nemo-streaming-fast-conformer-ctc-en-80ms"
        transducer = f"{d}/sherpa-onnx-nemo-streaming-fast-conformer-transducer-en-80ms"

        recognizers = []
        if Path(f"{ctc}/model.onnx").is_file():
            recognizers.append(
                sherpa_onnx.OnlineRecognizer.from_nemo_ctc(
                    model=f"{ctc}/model.onnx",
                    tokens=f"{ctc}/tokens.txt",
                    num_threads=1,
                )
            )
        if Path(f"{transducer}/encoder.onnx").is_file():
            recognizers.append(
                sherpa_onnx.OnlineRecognizer.from_transducer(
                    encoder=f"{transducer}/encoder.onnx",
                    decoder=f"{transducer}/decoder.onnx",
                    joiner=f"{transducer}/joiner.onnx",
                    tokens=f"{transducer}/tokens.txt",
                    num_threads=1,
                )
            )
        if not recognizers:
            print("skipping test_nemo_streaming_log_zero_guard()")
            return

        # Like NeMo, use log(x + 2**-24), not log(max(x, FLT_EPSILON)).
        # For digital silence, every feature is log(2**-24).
        for recognizer in recognizers:
            s = recognizer.create_stream()
            s.accept_waveform(16000, np.zeros(16000, dtype=np.float32))
            frames = np.array(s.get_frames(0, 10))
            np.testing.assert_allclose(frames, np.log(2.0**-24), rtol=0, atol=1e-5)

    def test_zipformer2_ctc(self):
        m = "sherpa-onnx-streaming-zipformer-ctc-multi-zh-hans-2023-12-13"
        for use_int8 in [True, False]:
            name = (
                "ctc-epoch-20-avg-1-chunk-16-left-128.int8.onnx"
                if use_int8
                else "ctc-epoch-20-avg-1-chunk-16-left-128.onnx"
            )
            model = f"{d}/{m}/{name}"
            tokens = f"{d}/{m}/tokens.txt"
            wave0 = f"{d}/{m}/test_wavs/DEV_T0000000000.wav"
            wave1 = f"{d}/{m}/test_wavs/DEV_T0000000001.wav"
            wave2 = f"{d}/{m}/test_wavs/DEV_T0000000002.wav"
            if not Path(model).is_file():
                print("skipping test_zipformer2_ctc()")
                return
            print(f"testing {model}")

            recognizer = sherpa_onnx.OnlineRecognizer.from_zipformer2_ctc(
                model=model,
                tokens=tokens,
                num_threads=1,
                provider="cpu",
            )

            streams = []
            waves = [wave0, wave1, wave2]
            for wave in waves:
                s = recognizer.create_stream()
                samples, sample_rate = read_wave(wave)
                s.accept_waveform(sample_rate, samples)

                tail_paddings = np.zeros(int(0.2 * sample_rate), dtype=np.float32)
                s.accept_waveform(sample_rate, tail_paddings)
                s.input_finished()
                streams.append(s)

            while True:
                ready_list = []
                for s in streams:
                    if recognizer.is_ready(s):
                        ready_list.append(s)
                if len(ready_list) == 0:
                    break
                recognizer.decode_streams(ready_list)

            results = [recognizer.get_result(s) for s in streams]
            for wave_filename, result in zip(waves, results):
                print(f"{wave_filename}\n{result}")
                print("-" * 10)

    def test_nemo_ctc_timestamps_and_endpoint(self):
        m = "sherpa-onnx-nemo-streaming-fast-conformer-ctc-en-80ms"
        model = f"{d}/{m}/model.onnx"
        tokens = f"{d}/{m}/tokens.txt"
        wave0 = f"{d}/{m}/test_wavs/0.wav"

        if not Path(model).is_file():
            print("skipping test_nemo_ctc_timestamps_and_endpoint()")
            return

        recognizer = sherpa_onnx.OnlineRecognizer.from_nemo_ctc(
            model=model,
            tokens=tokens,
            num_threads=1,
            enable_endpoint_detection=True,
            rule2_min_trailing_silence=1.2,
        )
        s = recognizer.create_stream()
        samples, sample_rate = read_wave(wave0)
        s.accept_waveform(sample_rate, samples)

        # The last word ends about 0.2 s before the end of the file
        silence = np.zeros(int(2 * sample_rate), dtype=np.float32)
        s.accept_waveform(sample_rate, silence)
        while recognizer.is_ready(s):
            recognizer.decode_stream(s)

        # The model has subsampling factor 8, so one output frame is 80 ms
        duration = len(samples) / sample_rate
        timestamps = recognizer.get_result_all(s).timestamps
        print(recognizer.get_result(s), timestamps[-1], duration)
        self.assertGreater(timestamps[-1], duration - 1)
        self.assertLess(timestamps[-1], duration)

        # About 2.2 s of trailing silence: rule 2 (1.2 s) must have fired
        self.assertTrue(recognizer.is_endpoint(s))

    def test_nemo_keep_encoder_states(self):
        ctc = f"{d}/sherpa-onnx-nemo-streaming-fast-conformer-ctc-en-80ms"
        transducer = f"{d}/sherpa-onnx-nemo-streaming-fast-conformer-transducer-en-80ms"

        tests = []
        if Path(f"{ctc}/model.onnx").is_file():
            recognizer = sherpa_onnx.OnlineRecognizer.from_nemo_ctc(
                model=f"{ctc}/model.onnx",
                tokens=f"{ctc}/tokens.txt",
                num_threads=1,
                enable_endpoint_detection=True,
            )
            # CTC has no decoder state: two utterances, 2 s of silence (rule 2)
            samples, sample_rate = read_wave(f"{ctc}/test_wavs/0.wav")
            silence = np.zeros(2 * sample_rate, dtype=np.float32)
            samples = np.concatenate([samples, silence, samples, silence])
            tests.append((recognizer, samples, sample_rate))
        if Path(f"{transducer}/encoder.onnx").is_file():
            recognizer = sherpa_onnx.OnlineRecognizer.from_transducer(
                encoder=f"{transducer}/encoder.onnx",
                decoder=f"{transducer}/decoder.onnx",
                joiner=f"{transducer}/joiner.onnx",
                tokens=f"{transducer}/tokens.txt",
                num_threads=1,
                enable_endpoint_detection=True,
            )
            # The decoder is still reset, so the endpoint has to come before
            # the first token: 3 s of leading silence (rule 1)
            samples, sample_rate = read_wave(f"{transducer}/test_wavs/0.wav")
            silence = np.zeros(3 * sample_rate, dtype=np.float32)
            samples = np.concatenate([silence, samples, silence[:sample_rate]])
            tests.append((recognizer, samples, sample_rate))
        if not tests:
            print("skipping test_nemo_keep_encoder_states()")
            return

        def decode(recognizer, samples, sample_rate, reset, keep=None):
            s = recognizer.create_stream()
            if keep is not None:
                s.set_option("keep_encoder_states", keep)
            s.accept_waveform(sample_rate, samples)
            results = []
            while recognizer.is_ready(s):
                recognizer.decode_stream(s)
                if reset and recognizer.is_endpoint(s):
                    results.append(recognizer.get_result_all(s))
                    recognizer.reset(s)
            results.append(recognizer.get_result_all(s))
            # (token, time in seconds from the start of the stream)
            tokens = [
                (token, round(r.start_time + t, 2))
                for r in results
                for token, t in zip(r.tokens, r.timestamps)
            ]
            return tokens, len(results)

        for recognizer, samples, sample_rate in tests:
            expected, _ = decode(recognizer, samples, sample_rate, reset=False)
            self.assertTrue(expected)

            # With the encoder states kept, the reset only splits the result
            kept, num_segments = decode(
                recognizer, samples, sample_rate, reset=True, keep="1"
            )
            self.assertGreater(num_segments, 1)
            self.assertEqual(kept, expected)

            # By default, the encoder states are reset
            default, _ = decode(recognizer, samples, sample_rate, reset=True)
            self.assertTrue(default)
            self.assertNotEqual(default, expected)

    def test_streaming_paraformer(self):
        m = "sherpa-onnx-streaming-paraformer-bilingual-zh-en"
        encoder = f"{d}/{m}/encoder.int8.onnx"
        decoder = f"{d}/{m}/decoder.int8.onnx"
        tokens = f"{d}/{m}/tokens.txt"
        wave2 = f"{d}/{m}/test_wavs/2.wav"

        if not Path(encoder).is_file():
            print("skipping test_streaming_paraformer()")
            return

        recognizer = sherpa_onnx.OnlineRecognizer.from_paraformer(
            encoder=encoder,
            decoder=decoder,
            tokens=tokens,
            num_threads=1,
            provider="cpu",
        )

        s = recognizer.create_stream()
        samples, sample_rate = read_wave(wave2)
        s.accept_waveform(sample_rate, samples)

        tail_paddings = np.zeros(int(0.66 * sample_rate), dtype=np.float32)
        s.accept_waveform(sample_rate, tail_paddings)
        s.input_finished()
        while recognizer.is_ready(s):
            recognizer.decode_stream(s)
        result = recognizer.get_result(s)
        print(f"{wave2}\n{result}")

        # The reference transcript is obtained with the feature frontend
        # the model was trained with (hamming window, snip_edges=True,
        # high_freq=0, see InitFeatConfig in
        # sherpa-onnx/csrc/online-recognizer-paraformer-impl.h); with the
        # default frontend, 频繁 is misrecognized as 平繁/苹繁.
        self.assertEqual(result, "这个是频繁的啊不认识接下来 frequently 频繁的")

    def test_wenet_ctc(self):
        models = [
            "sherpa-onnx-zh-wenet-aishell",
            "sherpa-onnx-zh-wenet-aishell2",
            "sherpa-onnx-zh-wenet-wenetspeech",
            "sherpa-onnx-zh-wenet-multi-cn",
            "sherpa-onnx-en-wenet-librispeech",
            "sherpa-onnx-en-wenet-gigaspeech",
        ]
        for m in models:
            for use_int8 in [True, False]:
                name = (
                    "model-streaming.int8.onnx" if use_int8 else "model-streaming.onnx"
                )
                model = f"{d}/{m}/{name}"
                tokens = f"{d}/{m}/tokens.txt"

                wave0 = f"{d}/{m}/test_wavs/0.wav"
                wave1 = f"{d}/{m}/test_wavs/1.wav"
                wave2 = f"{d}/{m}/test_wavs/8k.wav"

                if not Path(model).is_file():
                    print("skipping test_wenet_ctc()")
                    return

                recognizer = sherpa_onnx.OnlineRecognizer.from_wenet_ctc(
                    model=model,
                    tokens=tokens,
                    num_threads=1,
                    provider="cpu",
                )

                streams = []
                waves = [wave0, wave1, wave2]
                for wave in waves:
                    s = recognizer.create_stream()
                    samples, sample_rate = read_wave(wave)
                    s.accept_waveform(sample_rate, samples)

                    tail_paddings = np.zeros(int(0.2 * sample_rate), dtype=np.float32)
                    s.accept_waveform(sample_rate, tail_paddings)
                    s.input_finished()
                    streams.append(s)

                while True:
                    ready_list = []
                    for s in streams:
                        if recognizer.is_ready(s):
                            ready_list.append(s)
                    if len(ready_list) == 0:
                        break
                    recognizer.decode_streams(ready_list)

                results = [recognizer.get_result(s) for s in streams]
                for wave_filename, result in zip(waves, results):
                    print(f"{wave_filename}\n{result}")
                    print("-" * 10)


if __name__ == "__main__":
    unittest.main()
