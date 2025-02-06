#!/usr/bin/env python3
# Copyright    2025  Xiaomi Corp.        (authors: Fangjun Kuang)
import torch
from pathlib import Path


id2speaker = {
    0: "af_alloy",
    1: "af_aoede",
    2: "af_bella",
    3: "af_heart",
    4: "af_jessica",
    5: "af_kore",
    6: "af_nicole",
    7: "af_nova",
    8: "af_river",
    9: "af_sarah",
    10: "af_sky",
    11: "am_adam",
    12: "am_echo",
    13: "am_eric",
    14: "am_fenrir",
    15: "am_liam",
    16: "am_michael",
    17: "am_onyx",
    18: "am_puck",
    19: "am_santa",
    20: "bf_alice",
    21: "bf_emma",
    22: "bf_isabella",
    23: "bf_lily",
    24: "bm_daniel",
    25: "bm_fable",
    26: "bm_george",
    27: "bm_lewis",
    28: "ef_dora",
    29: "em_alex",
    30: "ff_siwis",
    31: "hf_alpha",
    32: "hf_beta",
    33: "hm_omega",
    34: "hm_psi",
    35: "if_sara",
    36: "im_nicola",
    37: "jf_alpha",
    38: "jf_gongitsune",
    39: "jf_nezumi",
    40: "jf_tebukuro",
    41: "jm_kumo",
    42: "pf_dora",
    43: "pm_alex",
    44: "pm_santa",
    45: "zf_xiaobei",
    46: "zf_xiaoni",
    47: "zf_xiaoxiao",
    48: "zf_xiaoyi",
    49: "zm_yunjian",
    50: "zm_yunxi",
    51: "zm_yunxia",
    52: "zm_yunyang",
}

speaker2id = {speaker: idx for idx, speaker in id2speaker.items()}


def main():
    if Path("./voices.bin").is_file():
        print("./voices.bin exists - skip")
        return

    with open("voices.bin", "wb") as f:
        for _, speaker in id2speaker.items():
            m = torch.load(
                f"voices/{speaker}.pt",
                weights_only=True,
                map_location="cpu",
            ).numpy()
            # m.shape (510, 1, 256)

            f.write(m.tobytes())


if __name__ == "__main__":
    main()
