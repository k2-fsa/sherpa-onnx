#!/usr/bin/env python3
# Copyright      2024  Xiaomi Corp.        (authors: Fangjun Kuang)

from typing import Dict

import onnx
import torch
import torchaudio
from nemo.collections.asr.models import EncDecRNNTBPEModel
from nemo.collections.asr.modules.audio_preprocessing import (
    AudioToMelSpectrogramPreprocessor as NeMoAudioToMelSpectrogramPreprocessor,
)
from nemo.collections.asr.parts.preprocessing.features import (
    FilterbankFeaturesTA as NeMoFilterbankFeaturesTA,
)
from onnxruntime.quantization import QuantType, quantize_dynamic


def add_meta_data(filename: str, meta_data: Dict[str, str]):
    """Add meta data to an ONNX model. It is changed in-place.

    Args:
      filename:
        Filename of the ONNX model to be changed.
      meta_data:
        Key-value pairs.
    """
    model = onnx.load(filename)
    while len(model.metadata_props):
        model.metadata_props.pop()

    for key, value in meta_data.items():
        meta = model.metadata_props.add()
        meta.key = key
        meta.value = str(value)

    onnx.save(model, filename)


class FilterbankFeaturesTA(NeMoFilterbankFeaturesTA):
    def __init__(self, mel_scale: str = "htk", wkwargs=None, **kwargs):
        if "window_size" in kwargs:
            del kwargs["window_size"]
        if "window_stride" in kwargs:
            del kwargs["window_stride"]

        super().__init__(**kwargs)

        self._mel_spec_extractor: torchaudio.transforms.MelSpectrogram = (
            torchaudio.transforms.MelSpectrogram(
                sample_rate=self._sample_rate,
                win_length=self.win_length,
                hop_length=self.hop_length,
                n_mels=kwargs["nfilt"],
                window_fn=self.torch_windows[kwargs["window"]],
                mel_scale=mel_scale,
                norm=kwargs["mel_norm"],
                n_fft=kwargs["n_fft"],
                f_max=kwargs.get("highfreq", None),
                f_min=kwargs.get("lowfreq", 0),
                wkwargs=wkwargs,
            )
        )


class AudioToMelSpectrogramPreprocessor(NeMoAudioToMelSpectrogramPreprocessor):
    def __init__(self, mel_scale: str = "htk", **kwargs):
        super().__init__(**kwargs)
        kwargs["nfilt"] = kwargs["features"]
        del kwargs["features"]
        self.featurizer = (
            FilterbankFeaturesTA(  # Deprecated arguments; kept for config compatibility
                mel_scale=mel_scale,
                **kwargs,
            )
        )


@torch.no_grad()
def main():
    model = EncDecRNNTBPEModel.from_config_file("./rnnt_model_config.yaml")
    ckpt = torch.load("./rnnt_model_weights.ckpt", map_location="cpu")
    model.load_state_dict(ckpt, strict=False)
    model.eval()

    with open("./tokens.txt", "w", encoding="utf-8") as f:
        for i, s in enumerate(model.joint.vocabulary):
            f.write(f"{s} {i}\n")
        f.write(f"<blk> {i+1}\n")
        print("Saved to tokens.txt")

    model.encoder.export("encoder.onnx")
    model.decoder.export("decoder.onnx")
    model.joint.export("joiner.onnx")

    meta_data = {
        "vocab_size": model.decoder.vocab_size,  # not including the blank
        "pred_rnn_layers": model.decoder.pred_rnn_layers,
        "pred_hidden": model.decoder.pred_hidden,
        "normalize_type": "",
        "subsampling_factor": 4,
        "model_type": "EncDecRNNTBPEModel",
        "version": "1",
        "model_author": "https://github.com/salute-developers/GigaAM",
        "license": "https://github.com/salute-developers/GigaAM/blob/main/GigaAM%20License_NC.pdf",
        "language": "Russian",
        "is_giga_am": 1,
    }
    add_meta_data("encoder.onnx", meta_data)

    quantize_dynamic(
        model_input="encoder.onnx",
        model_output="encoder.int8.onnx",
        weight_type=QuantType.QUInt8,
    )


if __name__ == "__main__":
    main()
