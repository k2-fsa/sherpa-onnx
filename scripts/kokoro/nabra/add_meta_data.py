#!/usr/bin/env python3
# Stamp sherpa-onnx Kokoro metadata onto a Nabra-82M ONNX model and bake in
# the iSTFT image-tone notch FIR.
#
# Usage: python3 add_meta_data.py <nabra.onnx> <output.onnx>
import sys

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper


def main():
    src, dst = sys.argv[1], sys.argv[2]

    meta = {
        "model_type": "kokoro",
        "language": "Arabic (Modern Standard)",
        "has_espeak": 1,
        "sample_rate": 24000,
        "version": 1,
        "voice": "ar",
        "style_dim": "510,1,256",
        "n_speakers": 1,
        "id2speaker": "0->af_msa",
        "speaker2id": "af_msa->0",
        "speaker_names": "af_msa",
        "max_token_len": 510,
        "comment": "Nabra-82M Arabic TTS (Kokoro architecture), base: oddadmix/Nabra-82M-v0.1",
    }

    m = onnx.load(src)
    while len(m.metadata_props):
        m.metadata_props.pop()
    for k, v in meta.items():
        e = m.metadata_props.add()
        e.key, e.value = k, str(v)

    # Bake iSTFT image-tone notch (4800 + 9600 Hz) as a final FIR Conv1d so
    # no player-side post-processing is required.
    g = m.graph
    out_name = g.output[0].name
    prod = {}
    for n in g.node:
        for o in n.output:
            prod[o] = n
    squeeze = prod[out_name]
    assert squeeze.op_type == "Squeeze", f"unexpected producer {squeeze.op_type}"
    src_tensor = squeeze.input[0]

    sr, ntaps = 24000, 129
    edges = [0, 4400, 4650, 4950, 5200, 9200, 9450, 9750, 10000, sr // 2]
    gains = [1, 1, 0, 0, 1, 1, 0, 0, 1, 0]
    from scipy.signal import firwin2
    h = firwin2(ntaps, np.array(edges) / (sr / 2), gains)

    conv = helper.make_node(
        "Conv",
        inputs=[src_tensor, "notch_fir_w"],
        outputs=["audio_notched"],
        kernel_shape=[ntaps],
        pads=[ntaps // 2] * 2,
        name="notch_fir",
    )
    g.initializer.append(
        numpy_helper.from_array(h.reshape(1, 1, -1).astype(np.float32),
                                "notch_fir_w"))
    for i, inp in enumerate(squeeze.input):
        if inp == src_tensor:
            squeeze.input[i] = "audio_notched"
    g.node.append(conv)

    onnx.save(m, dst)
    print(f"saved {dst}")


if __name__ == "__main__":
    main()
