# Copyright (c)  2023 by manyeyes
# Copyright (c)  2023  Xiaomi Corporation
from pathlib import Path
from typing import List, Optional

from _sherpa_onnx import (
    OfflineCtcFstDecoderConfig,
    OfflineFeatureExtractorConfig,
    OfflineModelConfig,
    OfflineNemoEncDecCtcModelConfig,
    OfflineParaformerModelConfig,
)
from _sherpa_onnx import OfflineRecognizer as _Recognizer
from _sherpa_onnx import (
    OfflineRecognizerConfig,
    OfflineStream,
    OfflineTdnnModelConfig,
    OfflineTransducerModelConfig,
    OfflineWenetCtcModelConfig,
    OfflineWhisperModelConfig,
    OfflineZipformerCtcModelConfig,
)


def _assert_file_exists(f: str):
    assert Path(f).is_file(), f"{f} does not exist"


class OfflineRecognizer(object):
    """A class for offline speech recognition.

    Please refer to the following files for usages
     - https://github.com/k2-fsa/sherpa-onnx/blob/master/sherpa-onnx/python/tests/test_offline_recognizer.py
     - https://github.com/k2-fsa/sherpa-onnx/blob/master/python-api-examples/offline-decode-files.py
    """

    @classmethod
    def from_transducer(
        cls,
        encoder: str,
        decoder: str,
        joiner: str,
        tokens: str,
        num_threads: int = 1,
        sample_rate: int = 16000,
        feature_dim: int = 80,
        decoding_method: str = "greedy_search",
        max_active_paths: int = 4,
        hotwords_file: str = "",
        hotwords_score: float = 1.5,
        blank_penalty: float = 0.0,
        debug: bool = False,
        provider: str = "cpu",
    ):
        """
        Please refer to
        `<https://k2-fsa.github.io/sherpa/onnx/pretrained_models/offline-transducer/index.html>`_
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
            Valid values: greedy_search, modified_beam_search.
          max_active_paths:
            Maximum number of active paths to keep. Used only when
            decoding_method is modified_beam_search.
          hotwords_file:
            The file containing hotwords, one words/phrases per line, and for each
            phrase the bpe/cjkchar are separated by a space.
          hotwords_score:
            The hotword score of each token for biasing word/phrase. Used only if
            hotwords_file is given with modified_beam_search as decoding method.
          blank_penalty:
            The penalty applied on blank symbol during decoding.
          debug:
            True to show debug messages.
          provider:
            onnxruntime execution providers. Valid values are: cpu, cuda, coreml.
        """
        self = cls.__new__(cls)
        model_config = OfflineModelConfig(
            transducer=OfflineTransducerModelConfig(
                encoder_filename=encoder,
                decoder_filename=decoder,
                joiner_filename=joiner,
            ),
            tokens=tokens,
            num_threads=num_threads,
            debug=debug,
            provider=provider,
            model_type="transducer",
        )

        feat_config = OfflineFeatureExtractorConfig(
            sampling_rate=sample_rate,
            feature_dim=feature_dim,
        )

        if len(hotwords_file) > 0 and decoding_method != "modified_beam_search":
            raise ValueError(
                "Please use --decoding-method=modified_beam_search when using "
                f"--hotwords-file. Currently given: {decoding_method}"
            )

        recognizer_config = OfflineRecognizerConfig(
            feat_config=feat_config,
            model_config=model_config,
            decoding_method=decoding_method,
            hotwords_file=hotwords_file,
            hotwords_score=hotwords_score,
            blank_penalty=blank_penalty,
        )
        self.recognizer = _Recognizer(recognizer_config)
        self.config = recognizer_config
        return self

    @classmethod
    def from_paraformer(
        cls,
        paraformer: str,
        tokens: str,
        num_threads: int = 1,
        sample_rate: int = 16000,
        feature_dim: int = 80,
        decoding_method: str = "greedy_search",
        debug: bool = False,
        provider: str = "cpu",
    ):
        """
        Please refer to
        `<https://k2-fsa.github.io/sherpa/onnx/pretrained_models/offline-paraformer/index.html>`_
        to download pre-trained models.

        Args:
          tokens:
            Path to ``tokens.txt``. Each line in ``tokens.txt`` contains two
            columns::

                symbol integer_id

          paraformer:
            Path to ``model.onnx``.
          num_threads:
            Number of threads for neural network computation.
          sample_rate:
            Sample rate of the training data used to train the model.
          feature_dim:
            Dimension of the feature used to train the model.
          decoding_method:
            Valid values are greedy_search.
          debug:
            True to show debug messages.
          provider:
            onnxruntime execution providers. Valid values are: cpu, cuda, coreml.
        """
        self = cls.__new__(cls)
        model_config = OfflineModelConfig(
            paraformer=OfflineParaformerModelConfig(model=paraformer),
            tokens=tokens,
            num_threads=num_threads,
            debug=debug,
            provider=provider,
            model_type="paraformer",
        )

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
        self.config = recognizer_config
        return self

    @classmethod
    def from_nemo_ctc(
        cls,
        model: str,
        tokens: str,
        num_threads: int = 1,
        sample_rate: int = 16000,
        feature_dim: int = 80,
        decoding_method: str = "greedy_search",
        debug: bool = False,
        provider: str = "cpu",
    ):
        """
        Please refer to
        `<https://k2-fsa.github.io/sherpa/onnx/pretrained_models/offline-ctc/nemo/index.html>`_
        to download pre-trained models for different languages, e.g., Chinese,
        English, etc.

        Args:
          model:
            Path to ``model.onnx``.
          tokens:
            Path to ``tokens.txt``. Each line in ``tokens.txt`` contains two
            columns::

                symbol integer_id

          num_threads:
            Number of threads for neural network computation.
          sample_rate:
            Sample rate of the training data used to train the model.
          feature_dim:
            Dimension of the feature used to train the model.
          decoding_method:
            Valid values are greedy_search.
          debug:
            True to show debug messages.
          provider:
            onnxruntime execution providers. Valid values are: cpu, cuda, coreml.
        """
        self = cls.__new__(cls)
        model_config = OfflineModelConfig(
            nemo_ctc=OfflineNemoEncDecCtcModelConfig(model=model),
            tokens=tokens,
            num_threads=num_threads,
            debug=debug,
            provider=provider,
            model_type="nemo_ctc",
        )

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
        self.config = recognizer_config
        return self

    @classmethod
    def from_whisper(
        cls,
        encoder: str,
        decoder: str,
        tokens: str,
        language: str = "en",
        task: str = "transcribe",
        num_threads: int = 1,
        decoding_method: str = "greedy_search",
        debug: bool = False,
        provider: str = "cpu",
        tail_paddings: int = -1,
    ):
        """
        Please refer to
        `<https://k2-fsa.github.io/sherpa/onnx/pretrained_models/whisper/index.html>`_
        to download pre-trained models for different kinds of whisper models,
        e.g., tiny, tiny.en, base, base.en, etc.

        Args:
          encoder_model:
            Path to the encoder model, e.g., tiny-encoder.onnx,
            tiny-encoder.int8.onnx, tiny-encoder.ort, etc.
          decoder_model:
            Path to the encoder model, e.g., tiny-encoder.onnx,
            tiny-encoder.int8.onnx, tiny-encoder.ort, etc.
          tokens:
            Path to ``tokens.txt``. Each line in ``tokens.txt`` contains two
            columns::

                symbol integer_id

          language:
            The spoken language in the audio file. Example values: en, de, zh,
            jp, fr. See https://github.com/openai/whisper/blob/main/whisper/tokenizer.py#L10
            for all possible values. Note that for non-multilingual models, the
            only valid value is 'en'.
          task:
            Valid values are: transcribe, translate. Note that for
            non-multilingual models, the only valid value is 'transcribe'.
          num_threads:
            Number of threads for neural network computation.
          decoding_method:
            Valid values: greedy_search.
          debug:
            True to show debug messages.
          provider:
            onnxruntime execution providers. Valid values are: cpu, cuda, coreml.
        """
        self = cls.__new__(cls)
        model_config = OfflineModelConfig(
            whisper=OfflineWhisperModelConfig(
                encoder=encoder,
                decoder=decoder,
                language=language,
                task=task,
                tail_paddings=tail_paddings,
            ),
            tokens=tokens,
            num_threads=num_threads,
            debug=debug,
            provider=provider,
            model_type="whisper",
        )

        feat_config = OfflineFeatureExtractorConfig(
            sampling_rate=16000,
            feature_dim=80,
        )

        recognizer_config = OfflineRecognizerConfig(
            feat_config=feat_config,
            model_config=model_config,
            decoding_method=decoding_method,
        )
        self.recognizer = _Recognizer(recognizer_config)
        self.config = recognizer_config
        return self

    @classmethod
    def from_tdnn_ctc(
        cls,
        model: str,
        tokens: str,
        num_threads: int = 1,
        sample_rate: int = 8000,
        feature_dim: int = 23,
        decoding_method: str = "greedy_search",
        debug: bool = False,
        provider: str = "cpu",
    ):
        """
        Please refer to
        `<https://k2-fsa.github.io/sherpa/onnx/pretrained_models/offline-ctc/yesno/index.html>`_
        to download pre-trained models.

        Args:
          model:
            Path to ``model.onnx``.
          tokens:
            Path to ``tokens.txt``. Each line in ``tokens.txt`` contains two
            columns::

                symbol integer_id

          num_threads:
            Number of threads for neural network computation.
          sample_rate:
            Sample rate of the training data used to train the model.
          feature_dim:
            Dimension of the feature used to train the model.
          decoding_method:
            Valid values are greedy_search.
          debug:
            True to show debug messages.
          provider:
            onnxruntime execution providers. Valid values are: cpu, cuda, coreml.
        """
        self = cls.__new__(cls)
        model_config = OfflineModelConfig(
            tdnn=OfflineTdnnModelConfig(model=model),
            tokens=tokens,
            num_threads=num_threads,
            debug=debug,
            provider=provider,
            model_type="tdnn",
        )

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
        self.config = recognizer_config
        return self

    @classmethod
    def from_wenet_ctc(
        cls,
        model: str,
        tokens: str,
        num_threads: int = 1,
        sample_rate: int = 16000,
        feature_dim: int = 80,
        decoding_method: str = "greedy_search",
        debug: bool = False,
        provider: str = "cpu",
    ):
        """
        Please refer to
        `<https://k2-fsa.github.io/sherpa/onnx/pretrained_models/whisper/index.html>`_
        to download pre-trained models for different languages, e.g., Chinese,
        English, etc.

        Args:
          model:
            Path to ``model.onnx``.
          tokens:
            Path to ``tokens.txt``. Each line in ``tokens.txt`` contains two
            columns::

                symbol integer_id

          num_threads:
            Number of threads for neural network computation.
          sample_rate:
            Sample rate of the training data used to train the model.
          feature_dim:
            Dimension of the feature used to train the model.
          decoding_method:
            Valid values are greedy_search.
          debug:
            True to show debug messages.
          provider:
            onnxruntime execution providers. Valid values are: cpu, cuda, coreml.
        """
        self = cls.__new__(cls)
        model_config = OfflineModelConfig(
            wenet_ctc=OfflineWenetCtcModelConfig(model=model),
            tokens=tokens,
            num_threads=num_threads,
            debug=debug,
            provider=provider,
            model_type="wenet_ctc",
        )

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
        self.config = recognizer_config
        return self

    def create_stream(self, hotwords: Optional[str] = None):
        if hotwords is None:
            return self.recognizer.create_stream()
        else:
            return self.recognizer.create_stream(hotwords)

    def decode_stream(self, s: OfflineStream):
        self.recognizer.decode_stream(s)

    def decode_streams(self, ss: List[OfflineStream]):
        self.recognizer.decode_streams(ss)
