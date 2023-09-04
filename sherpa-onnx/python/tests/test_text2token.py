# sherpa-onnx/python/tests/test_text2token.py
#
# Copyright (c)  2023  Xiaomi Corporation
#
# To run this single test, use
#
#  ctest --verbose -R  test_text2token_py

import unittest

import sherpa_onnx


class TestText2Token(unittest.TestCase):
    def test_bpe(self):
        texts = ["HELLO WORLD", "I LOVE YOU"]
        encoded_texts = sherpa_onnx.text2token(
            texts,
            tokens="testdata/tokens_en.txt",
            tokens_type="bpe",
            bpe_model="testdata/bpe_en.model",
        )
        assert encoded_texts == [
            ["▁HE", "LL", "O", "▁WORLD"],
            ["▁I", "▁LOVE", "▁YOU"],
        ], encoded_texts

        encoded_ids = sherpa_onnx.text2token(
            texts,
            tokens="testdata/tokens_en.txt",
            tokens_type="bpe",
            bpe_model="testdata/bpe_en.model",
            output_ids=True,
        )
        assert encoded_ids == [[22, 58, 24, 425], [19, 370, 47]], encoded_ids

    def test_cjkchar(self):
        texts = ["世界人民大团结", "中国 VS 美国"]
        encoded_texts = sherpa_onnx.text2token(
            texts, tokens="testdata/tokens_cn.txt", tokens_type="cjkchar"
        )
        assert encoded_texts == [
            ["世", "界", "人", "民", "大", "团", "结"],
            ["中", "国", "V", "S", "美", "国"],
        ], encoded_texts
        encoded_ids = sherpa_onnx.text2token(
            texts,
            tokens="testdata/tokens_cn.txt",
            tokens_type="cjkchar",
            output_ids=True,
        )
        assert encoded_ids == [
            [379, 380, 72, 874, 93, 1251, 489],
            [262, 147, 3423, 2476, 21, 147],
        ], encoded_ids

    def test_cjkchar_bpe(self):
        texts = ["世界人民 GOES TOGETHER", "中国 GOES WITH 美国"]
        encoded_texts = sherpa_onnx.text2token(
            texts,
            tokens="testdata/tokens_mix.txt",
            tokens_type="cjkchar+bpe",
            bpe_model="testdata/bpe_mix.model",
        )
        encoded_texts == [
            ["世", "界", "人", "民", "▁GO", "ES", "▁TOGETHER"],
            ["中", "国", "▁GO", "ES", "▁WITH", "美", "国"],
        ], encoded_texts
        encoded_ids = sherpa_onnx.text2token(
            texts,
            tokens="testdata/tokens_mix.txt",
            tokens_type="cjkchar+bpe",
            bpe_model="testdata/bpe_mix.model",
            output_ids=True,
        )
        encoded_ids == [
            [1368, 1392, 557, 680, 275, 178, 475],
            [685, 736, 275, 178, 179, 921, 736],
        ], encoded_ids


if __name__ == "__main__":
    unittest.main()
