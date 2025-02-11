#!/usr/bin/env python3
# This model exports the English-only TTS model.
# It has 5 speakers.
# {'EN-US': 0, 'EN-BR': 1, 'EN_INDIA': 2, 'EN-AU': 3, 'EN-Default': 4}

from typing import Any, Dict

import onnx
import torch
from melo.api import TTS
from melo.text import language_id_map, language_tone_start_map
from melo.text.chinese import pinyin_to_symbol_map
from melo.text.english import eng_dict, refine_syllables
from pypinyin import Style, lazy_pinyin, phrases_dict, pinyin_dict


def generate_tokens(symbol_list):
    with open("tokens.txt", "w", encoding="utf-8") as f:
        for i, s in enumerate(symbol_list):
            f.write(f"{s} {i}\n")


def add_new_english_words(lexicon):
    """
    Args:
      lexicon:
        Please modify it in-place.
    """

    # Please have a look at
    # https://github.com/myshell-ai/MeloTTS/blob/main/melo/text/cmudict.rep

    # We give several examples below about how to add new words

    # Example 1. Add a new word kaldi

    # It does not contain the word kaldi in cmudict.rep
    # so if we add the following line to cmudict.rep
    #
    #  KALDI K AH0 - L D IH0
    #
    # then we need to change the lexicon like below
    lexicon["kaldi"] = [["K", "AH0"], ["L", "D", "IH0"]]
    #
    # K AH0 and L D IH0 are separated by a dash "-", so
    # ["K", "AH0"] is a in list and ["L", "D", "IH0"] is in a separate list

    # Note: Either kaldi or KALDI is fine. You can use either lowercase or
    # uppercase or both

    # Example 2. Add a new word SF
    #
    # If we add the following line to cmudict.rep
    #
    #  SF EH1 S - EH1 F
    #
    # to cmudict.rep, then we need to change the lexicon like below:
    lexicon["SF"] = [["EH1", "S"], ["EH1", "F"]]

    # Please add your new words here

    # No need to return lexicon since it is changed in-place


def generate_lexicon():
    add_new_english_words(eng_dict)
    with open("lexicon.txt", "w", encoding="utf-8") as f:
        for word in eng_dict:
            phones, tones = refine_syllables(eng_dict[word])
            tones = [t + language_tone_start_map["EN"] for t in tones]
            tones = [str(t) for t in tones]

            phones = " ".join(phones)
            tones = " ".join(tones)

            f.write(f"{word.lower()} {phones} {tones}\n")


def add_meta_data(filename: str, meta_data: Dict[str, Any]):
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


class ModelWrapper(torch.nn.Module):
    def __init__(self, model: "SynthesizerTrn"):
        super().__init__()
        self.model = model
        self.lang_id = language_id_map[model.language]

    def forward(
        self,
        x,
        x_lengths,
        tones,
        sid,
        noise_scale,
        length_scale,
        noise_scale_w,
        max_len=None,
    ):
        """
        Args:
          x: A 1-D array of dtype np.int64. Its shape is (token_numbers,)
          tones: A 1-D array of dtype np.int64. Its shape is (token_numbers,)
          lang_id: A 1-D array of dtype np.int64. Its shape is (token_numbers,)
          sid: an integer
        """
        bert = torch.zeros(x.shape[0], 1024, x.shape[1], dtype=torch.float32)
        ja_bert = torch.zeros(x.shape[0], 768, x.shape[1], dtype=torch.float32)
        lang_id = torch.zeros_like(x)
        lang_id[:, 1::2] = self.lang_id
        return self.model.model.infer(
            x=x,
            x_lengths=x_lengths,
            sid=sid,
            tone=tones,
            language=lang_id,
            bert=bert,
            ja_bert=ja_bert,
            noise_scale=noise_scale,
            noise_scale_w=noise_scale_w,
            length_scale=length_scale,
        )[0]


def main():
    generate_lexicon()

    language = "EN"
    model = TTS(language=language, device="cpu")

    generate_tokens(model.hps["symbols"])

    torch_model = ModelWrapper(model)

    opset_version = 13
    x = torch.randint(low=0, high=10, size=(60,), dtype=torch.int64)
    print(x.shape)
    x_lengths = torch.tensor([x.size(0)], dtype=torch.int64)
    sid = torch.tensor([1], dtype=torch.int64)
    tones = torch.zeros_like(x)

    noise_scale = torch.tensor([1.0], dtype=torch.float32)
    length_scale = torch.tensor([1.0], dtype=torch.float32)
    noise_scale_w = torch.tensor([1.0], dtype=torch.float32)

    x = x.unsqueeze(0)
    tones = tones.unsqueeze(0)

    filename = "model.onnx"

    torch.onnx.export(
        torch_model,
        (
            x,
            x_lengths,
            tones,
            sid,
            noise_scale,
            length_scale,
            noise_scale_w,
        ),
        filename,
        opset_version=opset_version,
        input_names=[
            "x",
            "x_lengths",
            "tones",
            "sid",
            "noise_scale",
            "length_scale",
            "noise_scale_w",
        ],
        output_names=["y"],
        dynamic_axes={
            "x": {0: "N", 1: "L"},
            "x_lengths": {0: "N"},
            "tones": {0: "N", 1: "L"},
            "y": {0: "N", 1: "S", 2: "T"},
        },
    )

    meta_data = {
        "model_type": "melo-vits",
        "comment": "melo",
        "version": 2,
        "language": "English",
        "add_blank": int(model.hps.data.add_blank),
        "n_speakers": len(model.hps.data.spk2id),  # 5
        "jieba": 0,
        "sample_rate": model.hps.data.sampling_rate,
        "bert_dim": 1024,
        "ja_bert_dim": 768,
        "speaker_id": 0,
        "lang_id": language_id_map[model.language],
        "tone_start": language_tone_start_map[model.language],
        "url": "https://github.com/myshell-ai/MeloTTS",
        "license": "MIT license",
        "description": "MeloTTS is a high-quality multi-lingual text-to-speech library by MyShell.ai",
    }
    add_meta_data(filename, meta_data)


if __name__ == "__main__":
    main()
