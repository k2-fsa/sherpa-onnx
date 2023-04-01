# Copyright (c)  2023  Xiaomi Corporation
from pathlib import Path
from typing import List

from _sherpa_onnx import (
    OfflineFeatureExtractorConfig,
    OfflineRecognizer as _Recognizer,
    OfflineRecognizerConfig,
    OfflineStream,
    OfflineModelConfig,
    OfflineTransducerModelConfig,
    OfflineParaformerModelConfig,
)


def _assert_file_exists(f: str):
    assert Path(f).is_file(), f"{f} does not exist"


class OfflineRecognizer(object):
    """A class for offline speech recognition."""

    def __init__(
        self,
        tokens: str,
        encoder: str="",
        decoder: str="",
        joiner: str="",
        paraformer: str="",
        num_threads: int = 4,
        sample_rate: float = 16000,
        feature_dim: int = 80,
        decoding_method: str = "greedy_search",
        debug:bool=False,
    ):
        """
        Please refer to
        `<https://k2-fsa.github.io/sherpa/onnx/pretrained_models/index.html>`_
        to download pre-trained models for different languages, e.g., Chinese,
        English, etc.

        Args:
          tokens:
            Path to ``tokens.txt``. Each line in ``tokens.txt`` contains two
            columns::

                symbol integer_id

          encoder:
            Path to ``encoder.onnx``.
          decoder:
            Path to ``decoder.onnx``.
          joiner:
            Path to ``joiner.onnx``.
          paraformer:
            Path to ``paraformer.onnx``.
          num_threads:
            Number of threads for neural network computation.
          sample_rate:
            Sample rate of the training data used to train the model.
          feature_dim:
            Dimension of the feature used to train the model.
          enable_endpoint_detection:
            True to enable endpoint detection. False to disable endpoint
            detection.
          rule1_min_trailing_silence:
            Used only when enable_endpoint_detection is True. If the duration
            of trailing silence in seconds is larger than this value, we assume
            an endpoint is detected.
          rule2_min_trailing_silence:
            Used only when enable_endpoint_detection is True. If we have decoded
            something that is nonsilence and if the duration of trailing silence
            in seconds is larger than this value, we assume an endpoint is
            detected.
          rule3_min_utterance_length:
            Used only when enable_endpoint_detection is True. If the utterance
            length in seconds is larger than this value, we assume an endpoint
            is detected.
          decoding_method:
            Valid values are greedy_search, modified_beam_search.
          max_active_paths:
            Use only when decoding_method is modified_beam_search. It specifies
            the maximum number of active paths during beam search.
        """
        if len(encoder) > 0:
            _assert_file_exists(encoder)
            _assert_file_exists(decoder)
            _assert_file_exists(joiner)
        else:
            _assert_file_exists(paraformer)
        _assert_file_exists(tokens)

        assert num_threads > 0, num_threads

        model_config = OfflineModelConfig(
            transducer=OfflineTransducerModelConfig(
                encoder_filename="",
                decoder_filename="",
                joiner_filename=""
            ),
            paraformer=OfflineParaformerModelConfig(
                model=paraformer
            ),
            tokens=tokens,
            num_threads=num_threads,
            debug=debug
        )

        decoding_method = decoding_method

        feat_config = OfflineFeatureExtractorConfig(
            sampling_rate=sample_rate,
            feature_dim=feature_dim,
        )

        recognizer_config = OfflineRecognizerConfig(
            feat_config=feat_config,
            model_config=model_config,
            decoding_method=decoding_method,
        )

        self.recognizer = _Recognizer(recognizer_config)

    def create_stream(self):
        return self.recognizer.create_stream()

    def decode_stream(self, s: OfflineStream):
        self.recognizer.decode_stream(s)

    def decode_streams(self, ss: List[OfflineStream]):
        self.recognizer.decode_streams(ss)

    def get_result(self, s: OfflineStream) -> str:
        return s.result.text.strip()

    def get_results(self, ss: List[OfflineStream]) -> str:
        results = [s.result.text.strip() for s in ss]
        return results
