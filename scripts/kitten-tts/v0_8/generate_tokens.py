#!/usr/bin/env python3
# Copyright    2026  Xiaomi Corp.


def get_vocab():
    # https://github.com/KittenML/KittenTTS/blob/main/kittentts/onnx_model.py
    _pad = "$"
    _punctuation = ';:,.!?¡¿—…"«»"" '
    _letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
    _letters_ipa = "ɑɐɒæɓʙβɔɕçɗɖðʤəɘɚɛɜɝɞɟʄɡɠɢʛɦɧħɥʜɨɪʝɭɬɫɮʟɱɯɰŋɳɲɴøɵɸθœɶʘɹɺɾɻʀʁɽʂʃʈʧʉʊʋⱱʌɣɤʍχʎʏʑʐʒʔʡʕʢǀǁǂǃˈˌːˑʼʴʰʱʲʷˠˤ˞↓↑→↗↘'̩'ᵻ"

    symbols = [_pad, *_punctuation, *_letters, *_letters_ipa]
    return {symbols[i]: i for i in range(len(symbols))}


def main():
    token2id = get_vocab()
    with open("tokens.txt", "w", encoding="utf-8") as f:
        for s, i in token2id.items():
            f.write(f"{s} {i}\n")


if __name__ == "__main__":
    main()
