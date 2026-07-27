#!/usr/bin/env python3
# Copyright    2026  Xiaomi Corp.        (authors: Fangjun Kuang)


import time
from typing import List

import numpy as np
import onnxruntime as ort
import soundfile as sf

try:
    from piper_phonemize import phonemize_espeak
except Exception as ex:
    raise RuntimeError(
        f"{ex}\nPlease run\n"
        "pip install piper_phonemize -f https://k2-fsa.github.io/icefall/piper_phonemize.html"
    )


def get_token2id():
    _pad = "_"
    _punctuation = ';:,.!?¡¿—…"«»“” '
    _letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
    _letters_ipa = "ɑɐɒæɓʙβɔɕçɗɖðʤəɘɚɛɜɝɞɟʄɡɠɢʛɦɧħɥʜɨɪʝɭɬɫɮʟɱɯɰŋɳɲɴøɵɸθœɶʘɹɺɾɻʀʁɽʂʃʈʧʉʊʋⱱʌɣɤʍχʎʏʑʐʒʔʡʕʢǀǁǂǃˈˌːˑʼʴʰʱʲʷˠˤ˞↓↑→↗↘'̩'ᵻ"

    symbols = [_pad] + list(_punctuation) + list(_letters) + list(_letters_ipa)

    return {sym: idx for idx, sym in enumerate(symbols)}


def show(name, sess):
    print(f"----{name} input----")
    for i in sess.get_inputs():
        print(i)

    print(f"----{name} output----")

    for i in sess.get_outputs():
        print(i)

    print()


class OnnxModel:
    def __init__(self, duration_predictor: str, decoder: str):
        session_opts = ort.SessionOptions()
        session_opts.inter_op_num_threads = 1
        session_opts.intra_op_num_threads = 1

        self.session_opts = session_opts
        self.duration_predictor = ort.InferenceSession(
            duration_predictor,
            sess_options=self.session_opts,
            providers=["CPUExecutionProvider"],
        )

        self.decoder = ort.InferenceSession(
            decoder,
            sess_options=self.session_opts,
            providers=["CPUExecutionProvider"],
        )

        duration_predictor_meta_data = (
            self.duration_predictor.get_modelmeta().custom_metadata_map
        )
        decoder_meta_data = self.decoder.get_modelmeta().custom_metadata_map
        print("---duration_predictor_meta_data---")
        print(duration_predictor_meta_data)
        print("---decoder_meta_data---")
        print(decoder_meta_data)

        show("duration_predictor", self.duration_predictor)
        show("decoder", self.decoder)

        self.sample_rate = int(decoder_meta_data["sample_rate_hz"])

    def run_duration_predictor(self, tokens: List[int], speed=1.0):
        length_scale = np.array([1 / speed], dtype=np.float32)
        token_length = np.array([len(tokens)], dtype=np.int64)
        tokens = np.array(tokens, dtype=np.int64)[None]

        m_p_exp, logs_p_exp, y_mask = self.duration_predictor.run(
            [
                self.duration_predictor.get_outputs()[0].name,
                self.duration_predictor.get_outputs()[1].name,
                self.duration_predictor.get_outputs()[2].name,
            ],
            {
                self.duration_predictor.get_inputs()[0].name: tokens,
                self.duration_predictor.get_inputs()[1].name: token_length,
                self.duration_predictor.get_inputs()[2].name: length_scale,
            },
        )
        return m_p_exp, logs_p_exp, y_mask

    def run_decoder(self, m_p_exp, logs_p_exp, y_mask, noise, noise_scale=0.667):
        noise_scale = np.array([noise_scale], dtype=np.float32)
        waveform = self.decoder.run(
            [
                self.decoder.get_outputs()[0].name,
            ],
            {
                self.decoder.get_inputs()[0].name: m_p_exp,
                self.decoder.get_inputs()[1].name: logs_p_exp,
                self.decoder.get_inputs()[2].name: y_mask,
                self.decoder.get_inputs()[3].name: noise,
                self.decoder.get_inputs()[4].name: noise_scale,
            },
        )[0]
        return waveform


def main():

    token2id = get_token2id()
    with open("tokens.txt", "w", encoding="utf-8") as f:
        for s, i in token2id.items():
            f.write(f"{s} {i}\n")

    text = (
        "Today as always, men fall into two groups: slaves and free men."
        + " Whoever does not have two-thirds of his day for himself, "
        + "is a slave, whatever he may be: a statesman, a businessman, "
        + "an official, or a scholar."
    )

    model = OnnxModel(duration_predictor="./duration.onnx", decoder="./decode.onnx")
    start = time.time()

    tokens = phonemize_espeak(text, "en-us")
    #  print(text)
    #  print(tokens)
    tokens = sum(tokens, [])  # flatten
    #  print(tokens)

    ids = [token2id.get(t) for t in tokens]
    #  print(ids)
    padded = [0] * (2 * len(ids) + 1)
    padded[1::2] = ids
    #  print(padded)

    m_p_exp, logs_p_exp, y_mask = model.run_duration_predictor(padded)
    #  print(m_p_exp.shape, m_p_exp.dtype)  # (1, 128, 144)
    #  print(logs_p_exp.shape, logs_p_exp.dtype)  # (1, 128, 144)
    #  print(y_mask.shape, y_mask.dtype)  # (1, 1, 144)

    noise = np.random.randn(*m_p_exp.shape).astype(np.float32)
    #  print("noise.shape", noise.shape, noise.dtype)
    waveform = model.run_decoder(
        m_p_exp=m_p_exp,
        logs_p_exp=logs_p_exp,
        y_mask=y_mask,
        noise=noise,
    )

    waveform = waveform[0, 0]
    end = time.time()
    elapsed_seconds = end - start
    audio_duration = len(waveform) / model.sample_rate
    real_time_factor = elapsed_seconds / audio_duration

    sf.write(
        "generated.wav",
        waveform,
        samplerate=model.sample_rate,
        subtype="PCM_16",
    )

    print(f" Saved to ./generated.wav")
    print(f" Elapsed seconds: {elapsed_seconds:.3f}")
    print(f" Audio duration in seconds: {audio_duration:.3f}")
    print(f" RTF: {elapsed_seconds:.3f}/{audio_duration:.3f} = {real_time_factor:.3f}")


if __name__ == "__main__":
    main()
