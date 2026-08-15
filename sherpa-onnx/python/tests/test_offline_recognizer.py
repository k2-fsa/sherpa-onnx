# sherpa-onnx/python/tests/test_offline_recognizer.py
#
# Copyright (c)  2023  Xiaomi Corporation
#
# To run this single test, use
#
#  ctest --verbose -R  test_offline_recognizer_py

import unittest
import wave
from pathlib import Path
from typing import Tuple

import numpy as np
import sherpa_onnx

d = "/tmp/icefall-models"
# Please refer to
# https://k2-fsa.github.io/sherpa/onnx/pretrained_models/offline-transducer/index.html
# and
# https://k2-fsa.github.io/sherpa/onnx/pretrained_models/offline-paraformer/index.html
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


class TestOfflineRecognizer(unittest.TestCase):
    def test_from_transducer_positional_hotwords_compatibility(self):
        with self.assertRaisesRegex(ValueError, "hotwords-file"):
            sherpa_onnx.OfflineRecognizer.from_transducer(
                "encoder.onnx",
                "decoder.onnx",
                "joiner.onnx",
                "tokens.txt",
                1,
                16000,
                80,
                0.0,
                "greedy_search",
                4,
                "hotwords.txt",
            )

    def test_num_return_paths_validation(self):
        kwargs = {
            "encoder": "encoder.onnx",
            "decoder": "decoder.onnx",
            "joiner": "joiner.onnx",
            "tokens": "tokens.txt",
        }

        with self.assertRaisesRegex(ValueError, "range"):
            sherpa_onnx.OfflineRecognizer.from_transducer(
                **kwargs,
                decoding_method="modified_beam_search",
                max_active_paths=4,
                num_return_paths=5,
            )

        with self.assertRaisesRegex(ValueError, "modified_beam_search"):
            sherpa_onnx.OfflineRecognizer.from_transducer(
                **kwargs,
                decoding_method="greedy_search",
                num_return_paths=2,
            )

    def test_transducer_modified_beam_search_nbest(self):
        model_dir = Path(d) / "sherpa-onnx-zipformer-en-2023-04-01"
        encoder = model_dir / "encoder-epoch-99-avg-1.int8.onnx"
        decoder = model_dir / "decoder-epoch-99-avg-1.onnx"
        joiner = model_dir / "joiner-epoch-99-avg-1.int8.onnx"
        tokens = model_dir / "tokens.txt"
        wave0 = model_dir / "test_wavs" / "0.wav"

        if not encoder.is_file():
            print("skipping test_transducer_modified_beam_search_nbest()")
            return

        recognizer = sherpa_onnx.OfflineRecognizer.from_transducer(
            encoder=str(encoder),
            decoder=str(decoder),
            joiner=str(joiner),
            tokens=str(tokens),
            num_threads=1,
            decoding_method="modified_beam_search",
            max_active_paths=8,
            num_return_paths=3,
            provider="cpu",
        )

        stream = recognizer.create_stream()
        samples, sample_rate = read_wave(str(wave0))
        stream.accept_waveform(sample_rate, samples)
        recognizer.decode_stream(stream)

        hypotheses = stream.result.hypotheses
        self.assertGreaterEqual(len(hypotheses), 1)
        self.assertLessEqual(len(hypotheses), 3)
        self.assertEqual(stream.result.text, hypotheses[0].text)
        self.assertEqual(stream.result.tokens, hypotheses[0].tokens)
        self.assertTrue(
            all(
                hypotheses[i - 1].score >= hypotheses[i].score
                for i in range(1, len(hypotheses))
            )
        )

    def test_transducer_single_file(self):
        for use_int8 in [True, False]:
            if use_int8:
                encoder = f"{d}/sherpa-onnx-zipformer-en-2023-04-01/encoder-epoch-99-avg-1.int8.onnx"
                decoder = f"{d}/sherpa-onnx-zipformer-en-2023-04-01/decoder-epoch-99-avg-1.onnx"
                joiner = f"{d}/sherpa-onnx-zipformer-en-2023-04-01/joiner-epoch-99-avg-1.int8.onnx"
            else:
                encoder = f"{d}/sherpa-onnx-zipformer-en-2023-04-01/encoder-epoch-99-avg-1.onnx"
                decoder = f"{d}/sherpa-onnx-zipformer-en-2023-04-01/decoder-epoch-99-avg-1.onnx"
                joiner = f"{d}/sherpa-onnx-zipformer-en-2023-04-01/joiner-epoch-99-avg-1.onnx"

            tokens = f"{d}/sherpa-onnx-zipformer-en-2023-04-01/tokens.txt"
            wave0 = f"{d}/sherpa-onnx-zipformer-en-2023-04-01/test_wavs/0.wav"

            if not Path(encoder).is_file():
                print("skipping test_transducer_single_file()")
                return

            recognizer = sherpa_onnx.OfflineRecognizer.from_transducer(
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
            recognizer.decode_stream(s)
            print(s.result.text)

    def test_transducer_multiple_files(self):
        for use_int8 in [True, False]:
            if use_int8:
                encoder = f"{d}/sherpa-onnx-zipformer-en-2023-04-01/encoder-epoch-99-avg-1.int8.onnx"
                decoder = f"{d}/sherpa-onnx-zipformer-en-2023-04-01/decoder-epoch-99-avg-1.onnx"
                joiner = f"{d}/sherpa-onnx-zipformer-en-2023-04-01/joiner-epoch-99-avg-1.int8.onnx"
            else:
                encoder = f"{d}/sherpa-onnx-zipformer-en-2023-04-01/encoder-epoch-99-avg-1.onnx"
                decoder = f"{d}/sherpa-onnx-zipformer-en-2023-04-01/decoder-epoch-99-avg-1.onnx"
                joiner = f"{d}/sherpa-onnx-zipformer-en-2023-04-01/joiner-epoch-99-avg-1.onnx"

            tokens = f"{d}/sherpa-onnx-zipformer-en-2023-04-01/tokens.txt"
            wave0 = f"{d}/sherpa-onnx-zipformer-en-2023-04-01/test_wavs/0.wav"
            wave1 = f"{d}/sherpa-onnx-zipformer-en-2023-04-01/test_wavs/1.wav"
            wave2 = f"{d}/sherpa-onnx-zipformer-en-2023-04-01/test_wavs/8k.wav"

            if not Path(encoder).is_file():
                print("skipping test_transducer_multiple_files()")
                return

            recognizer = sherpa_onnx.OfflineRecognizer.from_transducer(
                encoder=encoder,
                decoder=decoder,
                joiner=joiner,
                tokens=tokens,
                num_threads=1,
                provider="cpu",
            )

            s0 = recognizer.create_stream()
            samples0, sample_rate0 = read_wave(wave0)
            s0.accept_waveform(sample_rate0, samples0)

            s1 = recognizer.create_stream()
            samples1, sample_rate1 = read_wave(wave1)
            s1.accept_waveform(sample_rate1, samples1)

            s2 = recognizer.create_stream()
            samples2, sample_rate2 = read_wave(wave2)
            s2.accept_waveform(sample_rate2, samples2)

            recognizer.decode_streams([s0, s1, s2])
            print(s0.result.text)
            print(s1.result.text)
            print(s2.result.text)

    def test_paraformer_single_file(self):
        model = f"{d}/sherpa-onnx-paraformer-zh-2023-09-14/model.int8.onnx"

        tokens = f"{d}/sherpa-onnx-paraformer-zh-2023-09-14/tokens.txt"
        wave0 = f"{d}/sherpa-onnx-paraformer-zh-2023-09-14/test_wavs/0.wav"

        if not Path(model).is_file():
            print("skipping test_paraformer_single_file()")
            return

        recognizer = sherpa_onnx.OfflineRecognizer.from_paraformer(
            paraformer=model,
            tokens=tokens,
            num_threads=1,
            provider="cpu",
        )

        s = recognizer.create_stream()
        samples, sample_rate = read_wave(wave0)
        s.accept_waveform(sample_rate, samples)
        recognizer.decode_stream(s)
        print(s.result.text)

    def test_paraformer_multiple_files(self):
        model = f"{d}/sherpa-onnx-paraformer-zh-2023-09-14/model.int8.onnx"

        tokens = f"{d}/sherpa-onnx-paraformer-zh-2023-09-14/tokens.txt"
        wave0 = f"{d}/sherpa-onnx-paraformer-zh-2023-09-14/test_wavs/0.wav"
        wave1 = f"{d}/sherpa-onnx-paraformer-zh-2023-09-14/test_wavs/1.wav"
        wave2 = f"{d}/sherpa-onnx-paraformer-zh-2023-09-14/test_wavs/2.wav"
        wave3 = f"{d}/sherpa-onnx-paraformer-zh-2023-09-14/test_wavs/8k.wav"

        if not Path(model).is_file():
            print("skipping test_paraformer_multiple_files()")
            return

        recognizer = sherpa_onnx.OfflineRecognizer.from_paraformer(
            paraformer=model,
            tokens=tokens,
            num_threads=1,
            provider="cpu",
        )

        s0 = recognizer.create_stream()
        samples0, sample_rate0 = read_wave(wave0)
        s0.accept_waveform(sample_rate0, samples0)

        s1 = recognizer.create_stream()
        samples1, sample_rate1 = read_wave(wave1)
        s1.accept_waveform(sample_rate1, samples1)

        s2 = recognizer.create_stream()
        samples2, sample_rate2 = read_wave(wave2)
        s2.accept_waveform(sample_rate2, samples2)

        s3 = recognizer.create_stream()
        samples3, sample_rate3 = read_wave(wave3)
        s3.accept_waveform(sample_rate3, samples3)

        recognizer.decode_streams([s0, s1, s2, s3])
        print(s0.result.text)
        print(s1.result.text)
        print(s2.result.text)
        print(s3.result.text)

    def test_nemo_ctc_single_file(self):
        for use_int8 in [True, False]:
            if use_int8:
                model = f"{d}/sherpa-onnx-nemo-ctc-en-citrinet-512/model.int8.onnx"
            else:
                model = f"{d}/sherpa-onnx-nemo-ctc-en-citrinet-512/model.onnx"

            tokens = f"{d}/sherpa-onnx-nemo-ctc-en-citrinet-512/tokens.txt"
            wave0 = f"{d}/sherpa-onnx-nemo-ctc-en-citrinet-512/test_wavs/0.wav"

            if not Path(model).is_file():
                print("skipping test_nemo_ctc_single_file()")
                return

            recognizer = sherpa_onnx.OfflineRecognizer.from_nemo_ctc(
                model=model,
                tokens=tokens,
                num_threads=1,
                provider="cpu",
            )

            s = recognizer.create_stream()
            samples, sample_rate = read_wave(wave0)
            s.accept_waveform(sample_rate, samples)
            recognizer.decode_stream(s)
            print(s.result.text)

    def test_nemo_ctc_multiple_files(self):
        for use_int8 in [True, False]:
            if use_int8:
                model = f"{d}/sherpa-onnx-nemo-ctc-en-citrinet-512/model.int8.onnx"
            else:
                model = f"{d}/sherpa-onnx-nemo-ctc-en-citrinet-512/model.onnx"

            tokens = f"{d}/sherpa-onnx-nemo-ctc-en-citrinet-512/tokens.txt"
            wave0 = f"{d}/sherpa-onnx-nemo-ctc-en-citrinet-512/test_wavs/0.wav"
            wave1 = f"{d}/sherpa-onnx-nemo-ctc-en-citrinet-512/test_wavs/1.wav"
            wave2 = f"{d}/sherpa-onnx-nemo-ctc-en-citrinet-512/test_wavs/8k.wav"

            if not Path(model).is_file():
                print("skipping test_nemo_ctc_multiple_files()")
                return

            recognizer = sherpa_onnx.OfflineRecognizer.from_nemo_ctc(
                model=model,
                tokens=tokens,
                num_threads=1,
                provider="cpu",
            )

            s0 = recognizer.create_stream()
            samples0, sample_rate0 = read_wave(wave0)
            s0.accept_waveform(sample_rate0, samples0)

            s1 = recognizer.create_stream()
            samples1, sample_rate1 = read_wave(wave1)
            s1.accept_waveform(sample_rate1, samples1)

            s2 = recognizer.create_stream()
            samples2, sample_rate2 = read_wave(wave2)
            s2.accept_waveform(sample_rate2, samples2)

            recognizer.decode_streams([s0, s1, s2])
            print(s0.result.text)
            print(s1.result.text)
            print(s2.result.text)

    def _test_wenet_ctc(self):
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
                name = "model.int8.onnx" if use_int8 else "model.onnx"
                model = f"{d}/{m}/{name}"
                tokens = f"{d}/{m}/tokens.txt"

                wave0 = f"{d}/{m}/test_wavs/0.wav"
                wave1 = f"{d}/{m}/test_wavs/1.wav"
                wave2 = f"{d}/{m}/test_wavs/8k.wav"

                if not Path(model).is_file():
                    print("skipping test_wenet_ctc()")
                    return

                recognizer = sherpa_onnx.OfflineRecognizer.from_wenet_ctc(
                    model=model,
                    tokens=tokens,
                    num_threads=1,
                    provider="cpu",
                )

                s0 = recognizer.create_stream()
                samples0, sample_rate0 = read_wave(wave0)
                s0.accept_waveform(sample_rate0, samples0)

                s1 = recognizer.create_stream()
                samples1, sample_rate1 = read_wave(wave1)
                s1.accept_waveform(sample_rate1, samples1)

                s2 = recognizer.create_stream()
                samples2, sample_rate2 = read_wave(wave2)
                s2.accept_waveform(sample_rate2, samples2)

                recognizer.decode_streams([s0, s1, s2])
                print(s0.result.text)
                print(s1.result.text)
                print(s2.result.text)


if __name__ == "__main__":
    unittest.main()
