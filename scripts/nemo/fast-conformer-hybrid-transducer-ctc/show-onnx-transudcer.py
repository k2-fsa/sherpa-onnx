#!/usr/bin/env python3

import onnxruntime

'''
encoder I/O

NodeArg(name='audio_signal', type='tensor(float)', shape=['audio_signal_dynamic_axes_1', 80, 'audio_signal_dynamic_axes_2'])
NodeArg(name='length', type='tensor(int64)', shape=['length_dynamic_axes_1'])
NodeArg(name='cache_last_channel', type='tensor(float)', shape=['cache_last_channel_dynamic_axes_1', 17, 'cache_last_channel_dynamic_axes_2', 512])
NodeArg(name='cache_last_time', type='tensor(float)', shape=['cache_last_time_dynamic_axes_1', 17, 512, 'cache_last_time_dynamic_axes_2'])
NodeArg(name='cache_last_channel_len', type='tensor(int64)', shape=['cache_last_channel_len_dynamic_axes_1'])
-----
NodeArg(name='outputs', type='tensor(float)', shape=['outputs_dynamic_axes_1', 512, 'outputs_dynamic_axes_2'])
NodeArg(name='encoded_lengths', type='tensor(int64)', shape=['encoded_lengths_dynamic_axes_1'])
NodeArg(name='cache_last_channel_next', type='tensor(float)', shape=['cache_last_channel_next_dynamic_axes_1', 17, 'cache_last_channel_next_dynamic_a
xes_2', 512])
NodeArg(name='cache_last_time_next', type='tensor(float)', shape=['cache_last_time_next_dynamic_axes_1', 17, 512, 'cache_last_time_next_dynamic_axes_
2'])
NodeArg(name='cache_last_channel_next_len', type='tensor(int64)', shape=['cache_last_channel_next_len_dynamic_axes_1'])
'''

'''
decoder I/O

NodeArg(name='encoder_outputs', type='tensor(float)', shape=['encoder_outputs_dynamic_axes_1', 512, 'encoder_outputs_dynamic_axes_2'])
NodeArg(name='targets', type='tensor(int32)', shape=['targets_dynamic_axes_1', 'targets_dynamic_axes_2'])
NodeArg(name='target_length', type='tensor(int32)', shape=['target_length_dynamic_axes_1'])
NodeArg(name='input_states_1', type='tensor(float)', shape=[1, 'input_states_1_dynamic_axes_1', 640])
NodeArg(name='input_states_2', type='tensor(float)', shape=[1, 'input_states_2_dynamic_axes_1', 640])
-----
NodeArg(name='outputs', type='tensor(float)', shape=['outputs_dynamic_axes_1', 'outputs_dynamic_axes_2', 'outputs_dynamic_axes_3', 1025])
NodeArg(name='prednet_lengths', type='tensor(int32)', shape=['prednet_lengths_dynamic_axes_1'])
NodeArg(name='output_states_1', type='tensor(float)', shape=[1, 'output_states_1_dynamic_axes_1', 640])
NodeArg(name='output_states_2', type='tensor(float)', shape=[1, 'output_states_2_dynamic_axes_1', 640])
'''

def show_encoder_io():
  session_opts = onnxruntime.SessionOptions()
  session_opts.log_severity_level = 3
  sess = onnxruntime.InferenceSession('encoder-model.onnx', session_opts)
  for i in sess.get_inputs():
    print(i)

  print('-----')


  for i in sess.get_outputs():
    print(i)

def show_decoder_io():
  session_opts = onnxruntime.SessionOptions()
  session_opts.log_severity_level = 3
  sess = onnxruntime.InferenceSession('decoder_joint-model.onnx', session_opts)
  for i in sess.get_inputs():
    print(i)

  print('-----')

  for i in sess.get_outputs():
    print(i)

def main():
  test_encoder()
  print('=========')
  test_decoder()

if __name__ == '__main__':
  main()
