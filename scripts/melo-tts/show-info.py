#!/usr/bin/env python3
# Copyright      2024  Xiaomi Corp.        (authors: Fangjun Kuang)

import onnxruntime


def show(filename):
    session_opts = onnxruntime.SessionOptions()
    session_opts.log_severity_level = 3
    sess = onnxruntime.InferenceSession(filename, session_opts)
    for i in sess.get_inputs():
        print(i)

    print("-----")

    for i in sess.get_outputs():
        print(i)

    meta = sess.get_modelmeta().custom_metadata_map
    print("*****************************************")
    print("meta\n", meta)


def main():
    print("=========model==========")
    show("./model.onnx")


if __name__ == "__main__":
    main()

"""
=========model==========
NodeArg(name='x', type='tensor(int64)', shape=['N', 'L'])
NodeArg(name='x_lengths', type='tensor(int64)', shape=['N'])
NodeArg(name='tones', type='tensor(int64)', shape=['N', 'L'])
NodeArg(name='sid', type='tensor(int64)', shape=[1])
NodeArg(name='noise_scale', type='tensor(float)', shape=[1])
NodeArg(name='length_scale', type='tensor(float)', shape=[1])
NodeArg(name='noise_scale_w', type='tensor(float)', shape=[1])
-----
NodeArg(name='y', type='tensor(float)', shape=['N', 'S', 'T'])
*****************************************
meta
 {'description': 'MeloTTS is a high-quality multi-lingual text-to-speech library by MyShell.ai',
 'model_type': 'melo-vits', 'license': 'MIT license', 'sample_rate': '44100', 'add_blank': '1',
 'n_speakers': '1', 'bert_dim': '1024', 'language': 'Chinese + English',
 'ja_bert_dim': '768', 'speaker_id': '1', 'comment': 'melo', 'lang_id': '3',
 'tone_start': '0', 'url': 'https://github.com/myshell-ai/MeloTTS'}
"""
