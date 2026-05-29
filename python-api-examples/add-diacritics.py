#!/usr/bin/env python3
#
# Copyright (c)  2026  Matias Lin

"""
This script shows how to add diacritics (tashkeel) to Arabic text
using the sherpa-onnx Python API.

It uses the CATT (Context-Aware Transformer for Tashkeel) Encoder-Only
checkpoint, which exports two ONNX files: an encoder and a decoder.

To use the CATT (Encoder-Only) diacritization model:

wget https://github.com/abjadai/catt/releases/download/v2/eo_model_onnx.zip
unzip eo_model_onnx.zip -d catt_eo_model_onnx
rm eo_model_onnx.zip
"""

from pathlib import Path

import sherpa_onnx


def main():
    catt_encoder = "./catt_eo_model_onnx/encoder.onnx"
    catt_decoder = "./catt_eo_model_onnx/decoder.onnx"

    for f in (catt_encoder, catt_decoder):
        if not Path(f).is_file():
            raise ValueError(f"{f} does not exist")

    config = sherpa_onnx.OfflineDiacritizationConfig(
        model=sherpa_onnx.OfflineDiacritizationModelConfig(
            catt_encoder=catt_encoder,
            catt_decoder=catt_decoder,
        ),
    )

    diacrt = sherpa_onnx.OfflineDiacritization(config)

    text_list = [
        "وقالت مجلة نيوزويك الأمريكية التحديث الجديد ل إنستجرام يمكن أن يساهم في إيقاف وكشف الحسابات المزورة بسهولة شديدة",
        "اللغة العربية من أقدم اللغات السامية",
    ]
    for text in text_list:
        text_with_diacritics = diacrt.add_diacritics(text)
        print("----------")
        print(f"input:  {text}")
        print(f"output: {text_with_diacritics}")

    print("----------")


if __name__ == "__main__":
    main()
