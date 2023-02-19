from pathlib import Path
from typing import List

from _sherpa_onnx import (
    OnlineStream,
    OnlineTransducerModelConfig,
    FeatureExtractorConfig,
    OnlineRecognizerConfig,
)
from _sherpa_onnx import OnlineRecognizer as _Recognizer


def _assert_file_exists(f: str):
    assert Path(f).is_file(), f"{f} does not exist"


class OnlineRecognizer(object):
    """A class for streaming speech recognition."""

    def __init__(
        self,
        tokens: str,
        encoder: str,
        decoder: str,
        joiner: str,
        num_threads: int = 4,
        sample_rate: float = 16000,
        feature_dim: int = 80,
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
            num_threads=num_threads,
        )

        feat_config = FeatureExtractorConfig(
            sampling_rate=sample_rate,
            feature_dim=feature_dim,
        )

        recognizer_config = OnlineRecognizerConfig(
            feat_config=feat_config,
            model_config=model_config,
            tokens=tokens,
        )

        self.recognizer = _Recognizer(recognizer_config)

    def create_stream(self):
        return self.recognizer.create_stream()

    def decode_stream(self, s: OnlineStream):
        self.recognizer.decode_stream(s)

    def decode_streams(self, ss: List[OnlineStream]):
        self.recognizer.decode_streams(ss)

    def is_ready(self, s: OnlineStream) -> bool:
        return self.recognizer.is_ready(s)

    def get_result(self, s: OnlineStream) -> str:
        return self.recognizer.get_result(s).text
