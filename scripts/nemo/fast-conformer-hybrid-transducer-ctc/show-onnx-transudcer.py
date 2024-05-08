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


def main():
    print("=========encoder==========")
    show("./encoder.onnx")

    print("=========decoder==========")
    show("./decoder.onnx")

    print("=========joiner==========")
    show("./joiner.onnx")


if __name__ == "__main__":
    main()

"""
=========encoder==========
NodeArg(name='audio_signal', type='tensor(float)', shape=['audio_signal_dynamic_axes_1', 80, 'audio_signal_dynamic_axes_2'])
NodeArg(name='length', type='tensor(int64)', shape=['length_dynamic_axes_1'])
NodeArg(name='cache_last_channel', type='tensor(float)', shape=['cache_last_channel_dynamic_axes_1', 17, 'cache_last_channel_dynamic_axes_2', 512])
NodeArg(name='cache_last_time', type='tensor(float)', shape=['cache_last_time_dynamic_axes_1', 17, 512, 'cache_last_time_dynamic_axes_2'])
NodeArg(name='cache_last_channel_len', type='tensor(int64)', shape=['cache_last_channel_len_dynamic_axes_1'])
-----
NodeArg(name='outputs', type='tensor(float)', shape=['outputs_dynamic_axes_1', 512, 'outputs_dynamic_axes_2'])
NodeArg(name='encoded_lengths', type='tensor(int64)', shape=['encoded_lengths_dynamic_axes_1'])
NodeArg(name='cache_last_channel_next', type='tensor(float)', shape=['cache_last_channel_next_dynamic_axes_1', 17, 'cache_last_channel_next_dynamic_axes_2', 512])
NodeArg(name='cache_last_time_next', type='tensor(float)', shape=['cache_last_time_next_dynamic_axes_1', 17, 512, 'cache_last_time_next_dynamic_axes_2'])
NodeArg(name='cache_last_channel_next_len', type='tensor(int64)', shape=['cache_last_channel_next_len_dynamic_axes_1'])
=========decoder==========
NodeArg(name='targets', type='tensor(int32)', shape=['targets_dynamic_axes_1', 'targets_dynamic_axes_2'])
NodeArg(name='target_length', type='tensor(int32)', shape=['target_length_dynamic_axes_1'])
NodeArg(name='states.1', type='tensor(float)', shape=[1, 'states.1_dim_1', 640])
NodeArg(name='onnx::LSTM_3', type='tensor(float)', shape=[1, 1, 640])
-----
NodeArg(name='outputs', type='tensor(float)', shape=['outputs_dynamic_axes_1', 640, 'outputs_dynamic_axes_2'])
NodeArg(name='prednet_lengths', type='tensor(int32)', shape=['prednet_lengths_dynamic_axes_1'])
NodeArg(name='states', type='tensor(float)', shape=[1, 'states_dynamic_axes_1', 640])
NodeArg(name='74', type='tensor(float)', shape=[1, 'LSTM74_dim_1', 640])
=========joiner==========
NodeArg(name='encoder_outputs', type='tensor(float)', shape=['encoder_outputs_dynamic_axes_1', 512, 'encoder_outputs_dynamic_axes_2'])
NodeArg(name='decoder_outputs', type='tensor(float)', shape=['decoder_outputs_dynamic_axes_1', 640, 'decoder_outputs_dynamic_axes_2'])
-----
NodeArg(name='outputs', type='tensor(float)', shape=['outputs_dynamic_axes_1', 'outputs_dynamic_axes_2', 'outputs_dynamic_axes_3', 1025])

"""
