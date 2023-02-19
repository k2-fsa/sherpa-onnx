# sherpa-onnx/python/tests/test_online_transducer_model_config.py
#
# Copyright (c)  2023  Xiaomi Corporation
#
# To run this single test, use
#
#  ctest --verbose -R  test_online_transducer_model_config_py

import unittest

import sherpa_onnx


class TestOnlineTransducerModelConfig(unittest.TestCase):
    def test_constructor(self):
        config = sherpa_onnx.OnlineTransducerModelConfig(
            encoder_filename="encoder.onnx",
            decoder_filename="decoder.onnx",
            joiner_filename="joiner.onnx",
            num_threads=8,
            debug=True,
        )
        assert config.encoder_filename == "encoder.onnx", config.encoder_filename
        assert config.decoder_filename == "decoder.onnx", config.decoder_filename
        assert config.joiner_filename == "joiner.onnx", config.joiner_filename
        assert config.num_threads == 8, config.num_threads
        assert config.debug is True, config.debug
        print(config)


if __name__ == "__main__":
    unittest.main()
