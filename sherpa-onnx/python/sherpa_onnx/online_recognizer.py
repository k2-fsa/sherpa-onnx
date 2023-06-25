# Copyright (c)  2023  Xiaomi Corporation
from pathlib import Path
from typing import List, Optional

from _sherpa_onnx import (
    EndpointConfig,
    FeatureExtractorConfig,
    OnlineRecognizer as _Recognizer,
    OnlineRecognizerConfig,
    OnlineStream,
    OnlineTransducerModelConfig,
)


def _assert_file_exists(f: str):
    assert Path(f).is_file(), f"{f} does not exist"


class OnlineRecognizer(object):
    """A class for streaming speech recognition.

    Please refer to the following files for usages
     - https://github.com/k2-fsa/sherpa-onnx/blob/master/sherpa-onnx/python/tests/test_online_recognizer.py
     - https://github.com/k2-fsa/sherpa-onnx/blob/master/python-api-examples/online-decode-files.py
    """

    def __init__(
        self,
        tokens: str,
        encoder: str,
        decoder: str,
        joiner: str,
        num_threads: int = 4,
        sample_rate: float = 16000,
        feature_dim: int = 80,
        enable_endpoint_detection: bool = False,
        rule1_min_trailing_silence: float = 2.4,
        rule2_min_trailing_silence: float = 1.2,
        rule3_min_utterance_length: float = 20.0,
        decoding_method: str = "greedy_search",
        max_active_paths: int = 4,
        context_score: float = 1.5,
        provider: str = "cpu",
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
          provider:
            onnxruntime execution providers. Valid values are: cpu, cuda, coreml.
        """
        _assert_file_exists(tokens)
        _assert_file_exists(encoder)
        _assert_file_exists(decoder)
        _assert_file_exists(joiner)

        assert num_threads > 0, num_threads

        model_config = OnlineTransducerModelConfig(
            encoder_filename=encoder,
            decoder_filename=decoder,
            joiner_filename=joiner,
            tokens=tokens,
            num_threads=num_threads,
            provider=provider,
        )

        feat_config = FeatureExtractorConfig(
            sampling_rate=sample_rate,
            feature_dim=feature_dim,
        )

        endpoint_config = EndpointConfig(
            rule1_min_trailing_silence=rule1_min_trailing_silence,
            rule2_min_trailing_silence=rule2_min_trailing_silence,
            rule3_min_utterance_length=rule3_min_utterance_length,
        )

        recognizer_config = OnlineRecognizerConfig(
            feat_config=feat_config,
            model_config=model_config,
            endpoint_config=endpoint_config,
            enable_endpoint=enable_endpoint_detection,
            decoding_method=decoding_method,
            max_active_paths=max_active_paths,
            context_score=context_score,
        )

        self.recognizer = _Recognizer(recognizer_config)
        self.config = recognizer_config

    def create_stream(self, contexts_list : Optional[List[List[int]]] = None):
        if contexts_list is None:
            return self.recognizer.create_stream()
        else:
            return self.recognizer.create_stream(contexts_list)

    def decode_stream(self, s: OnlineStream):
        self.recognizer.decode_stream(s)

    def decode_streams(self, ss: List[OnlineStream]):
        self.recognizer.decode_streams(ss)

    def is_ready(self, s: OnlineStream) -> bool:
        return self.recognizer.is_ready(s)

    def get_result(self, s: OnlineStream) -> str:
        return self.recognizer.get_result(s).text.strip()

    def is_endpoint(self, s: OnlineStream) -> bool:
        return self.recognizer.is_endpoint(s)

    def reset(self, s: OnlineStream) -> bool:
        return self.recognizer.reset(s)
