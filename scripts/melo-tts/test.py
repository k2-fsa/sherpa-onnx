#!/usr/bin/env python3

from typing import Iterable, List, Tuple

import jieba
import onnxruntime as ort
import soundfile as sf
import torch


class Lexicon:
    def __init__(self, lexion_filename: str, tokens_filename: str):
        tokens = dict()
        with open(tokens_filename, encoding="utf-8") as f:
            for line in f:
                s, i = line.split()
                tokens[s] = int(i)

        lexicon = dict()
        with open(lexion_filename, encoding="utf-8") as f:
            for line in f:
                splits = line.split()
                word_or_phrase = splits[0]
                phone_tone_list = splits[1:]
                assert len(phone_tone_list) & 1 == 0, len(phone_tone_list)
                phones = phone_tone_list[: len(phone_tone_list) // 2]
                phones = [tokens[p] for p in phones]

                tones = phone_tone_list[len(phone_tone_list) // 2 :]
                tones = [int(t) for t in tones]

                lexicon[word_or_phrase] = (phones, tones)
        self.lexicon = lexicon

        punctuation = ["!", "?", "…", ",", ".", "'", "-"]
        for p in punctuation:
            i = tokens[p]
            tone = 0
            self.lexicon[p] = ([i], [tone])
        self.lexicon[" "] = ([tokens["_"]], [0])

    def _convert(self, text: str) -> Tuple[List[int], List[int]]:
        phones = []
        tones = []

        if text == "，":
            text = ","
        elif text == "。":
            text = "."
        elif text == "！":
            text = "!"
        elif text == "？":
            text = "?"

        if text not in self.lexicon:
            print("t", text)
            if len(text) > 1:
                for w in text:
                    print("w", w)
                    p, t = self.convert(w)
                    if p:
                        phones += p
                        tones += t
            return phones, tones

        phones, tones = self.lexicon[text]
        return phones, tones

    def convert(self, text_list: Iterable[str]) -> Tuple[List[int], List[int]]:
        phones = []
        tones = []
        for text in text_list:
            print(text)
            p, t = self._convert(text)
            phones += p
            tones += t
        return phones, tones


class OnnxModel:
    def __init__(self, filename):
        session_opts = ort.SessionOptions()
        session_opts.inter_op_num_threads = 1
        session_opts.intra_op_num_threads = 4

        self.session_opts = session_opts
        self.model = ort.InferenceSession(
            filename,
            sess_options=self.session_opts,
            providers=["CPUExecutionProvider"],
        )
        meta = self.model.get_modelmeta().custom_metadata_map
        self.bert_dim = int(meta["bert_dim"])
        self.ja_bert_dim = int(meta["ja_bert_dim"])
        self.add_blank = int(meta["add_blank"])
        self.sample_rate = int(meta["sample_rate"])
        self.speaker_id = int(meta["speaker_id"])
        self.lang_id = int(meta["lang_id"])
        self.sample_rate = int(meta["sample_rate"])

    def __call__(self, x, tones, lang):
        """
        Args:
          x: 1-D int64 torch tensor
          tones: 1-D int64 torch tensor
          lang: 1-D int64 torch tensor
        """
        x = x.unsqueeze(0)
        tones = tones.unsqueeze(0)
        lang = lang.unsqueeze(0)

        print(x.shape, tones.shape, lang.shape)
        bert = torch.zeros(1, self.bert_dim, x.shape[-1])
        ja_bert = torch.zeros(1, self.ja_bert_dim, x.shape[-1])
        sid = torch.tensor([self.speaker_id], dtype=torch.int64)
        noise_scale = torch.tensor([0.6], dtype=torch.float32)
        length_scale = torch.tensor([1.0], dtype=torch.float32)
        noise_scale_w = torch.tensor([0.8], dtype=torch.float32)

        x_lengths = torch.tensor([x.shape[-1]], dtype=torch.int64)

        y = self.model.run(
            ["y"],
            {
                "x": x.numpy(),
                "x_lengths": x_lengths.numpy(),
                "tones": tones.numpy(),
                "lang_id": lang.numpy(),
                "bert": bert.numpy(),
                "ja_bert": ja_bert.numpy(),
                "sid": sid.numpy(),
                "noise_scale": noise_scale.numpy(),
                "noise_scale_w": noise_scale_w.numpy(),
                "length_scale": length_scale.numpy(),
            },
        )[0][0][0]
        return y


def main():
    lexicon = Lexicon(lexion_filename="./lexicon.txt", tokens_filename="./tokens.txt")

    text = "永远相信，美好的事情即将发生。多音字测试， 银行，行不行？长沙长大"
    s = jieba.cut(text, HMM=True)

    phones, tones = lexicon.convert(s)

    model = OnnxModel("./model.onnx")
    langs = [model.lang_id] * len(phones)

    if model.add_blank:
        new_phones = [0] * (2 * len(phones) + 1)
        new_tones = [0] * (2 * len(tones) + 1)
        new_langs = [0] * (2 * len(langs) + 1)

        new_phones[1::2] = phones
        new_tones[1::2] = tones
        new_langs[1::2] = langs

        phones = new_phones
        tones = new_tones
        langs = new_langs

    phones = torch.tensor(phones, dtype=torch.int64)
    tones = torch.tensor(tones, dtype=torch.int64)
    langs = torch.tensor(langs, dtype=torch.int64)

    print(phones.shape, tones.shape, langs.shape)

    y = model(x=phones, tones=tones, lang=langs)
    sf.write("./test.wav", y, model.sample_rate)


if __name__ == "__main__":
    main()
