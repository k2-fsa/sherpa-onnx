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

    @classmethod
    def from_transducer(
            cls,
            encoder: str,
            decoder: str,
            joiner: str,
            tokens: str,
            num_threads: int,
            sample_rate: int = 16000,
            feature_dim: int = 80,
            decoding_method: str = "greedy_search",
            debug: bool = False,
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
          decoding_method:
            Valid values are greedy_search, modified_beam_search.
          debug:
            True to show debug messages.
        """
        self = cls.__new__(cls)
        model_config = OfflineModelConfig(
            transducer=OfflineTransducerModelConfig(
                encoder_filename=encoder,
                decoder_filename=decoder,
                joiner_filename=joiner
            ),
            paraformer=OfflineParaformerModelConfig(
                model=""
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
        print("config ok!")
        self.recognizer = _Recognizer(recognizer_config)
        return self

    @classmethod
    def from_paraformer(
            cls,
            paraformer: str,
            tokens: str,
            num_threads: int,
            sample_rate: int = 16000,
            feature_dim: int = 80,
            decoding_method: str = "greedy_search",
            debug: bool = False,
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

          paraformer:
            Path to ``paraformer.onnx``.
          num_threads:
            Number of threads for neural network computation.
          sample_rate:
            Sample rate of the training data used to train the model.
          feature_dim:
            Dimension of the feature used to train the model.
          decoding_method:
            Valid values are greedy_search, modified_beam_search.
          debug:
            True to show debug messages.
        """
        self = cls.__new__(cls)
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
        print("config ok!")
        self.recognizer = _Recognizer(recognizer_config)
        return self

    def create_stream(self):
        return self.recognizer.create_stream()

    def decode_stream(self, s: OfflineStream):
        self.recognizer.decode_stream(s)

    def decode_streams(self, ss: List[OfflineStream]):
        self.recognizer.decode_streams(ss)

