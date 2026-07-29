# sherpa-onnx/python/tests/test_offline_tts_config.py
#
# Copyright (c)  2026  Xiaomi Corporation
#
# To run this single test, use
#
#  ctest --verbose -R  test_offline_tts_config_py

import tempfile
import unittest
from pathlib import Path

import _sherpa_onnx


def _touch(path: Path):
    path.write_text("dummy")


def _create_kokoro_files(d: Path):
    _touch(d / "model.onnx")
    _touch(d / "voices.bin")
    _touch(d / "tokens.txt")

    data_dir = d / "espeak-ng-data"
    data_dir.mkdir()
    for name in ["phontab", "phonindex", "phondata", "intonations"]:
        _touch(data_dir / name)


def _make_kokoro_config(d: Path, **kwargs):
    args = {
        "model": str(d / "model.onnx"),
        "voices": str(d / "voices.bin"),
        "tokens": str(d / "tokens.txt"),
        "data_dir": str(d / "espeak-ng-data"),
    }
    args.update(kwargs)
    return _sherpa_onnx.OfflineTtsKokoroModelConfig(**args)


class TestOfflineTtsKokoroModelConfig(unittest.TestCase):
    def test_all_files_exist(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            _create_kokoro_files(d)
            config = _make_kokoro_config(d)
            assert config.validate(), str(config)

    def test_missing_voices_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            _create_kokoro_files(d)
            config = _make_kokoro_config(d, voices=str(d / "not-exist.bin"))
            assert not config.validate(), str(config)

    def test_empty_voices(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            _create_kokoro_files(d)
            config = _make_kokoro_config(d, voices="")
            assert not config.validate(), str(config)

    def test_missing_model_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            _create_kokoro_files(d)
            config = _make_kokoro_config(d, model=str(d / "not-exist.onnx"))
            assert not config.validate(), str(config)


if __name__ == "__main__":
    unittest.main()
