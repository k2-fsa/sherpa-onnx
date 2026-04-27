# Introduction

This folder contains scripts to export the model from
<https://huggingface.co/nvidia/parakeet-unified-en-0.6b>
to sherpa-onnx.

## asr_model.cfg

```
{'sample_rate': 16000, 'compute_eval_loss': False, 'log_prediction': True,
'rnnt_reduction': 'mean_volume', 'skip_nan_grad': False,
'model_defaults': {'enc_hidden': 1024, 'pred_hidden': 640, 'joint_hidden': 640},
'train_ds': {'use_lhotse': True, 'skip_missing_manifest_entries': True, 'input_cfg': None,
'manifest_filepath': None, 'sample_rate': 16000, 'batch_size': 16, 'shuffle': True,
'num_workers': 8, 'pin_memory': True, 'max_duration': 40.0, 'min_duration': 0.1,
'text_field': 'answer', 'batch_duration': None, 'use_bucketing': True,
'max_tps': None, 'bucket_duration_bins': None, 'bucket_batch_size': None,
'num_buckets': None, 'bucket_buffer_size': None, 'shuffle_buffer_size': None, 'tarred_audio_filepaths': None,
'augmentor': None}, 'validation_ds': {},
'tokenizer': {'dir': None, 'type': 'bpe', 'model_path': 'nemo:c9e35cde64e14bdc87cf70d543842217_tokenizer.model',
  'vocab_path': 'nemo:28f042954ba747e99209b8ca5a223ba3_vocab.txt', 'spe_tokenizer_vocab': 'nemo:aa68b93b03344274b0c0e2a96333de24_tokenizer.vocab'},
'preprocessor': {'_target_': 'nemo.collections.asr.modules.AudioToMelSpectrogramPreprocessor',
  'sample_rate': 16000, 'normalize': 'per_feature', 'window_size': 0.025, 'window_stride': 0.01,
  'window': 'hann', 'features': 128, 'n_fft': 512, 'frame_splicing': 1,
  'dither': 1e-05, 'pad_to': 0, 'pad_value': 0.0},
'spec_augment': {'_target_': 'nemo.collections.asr.modules.SpectrogramAugmentation', 'freq_masks': 2, 'time_masks': 10, 'freq_width': 27, 'time_width': 0.05},
'encoder': {'_target_': 'nemo.collections.asr.modules.ConformerEncoder',
  'feat_in': 128, 'feat_out': -1, 'n_layers': 24, 'd_model': 1024,
  'subsampling': 'dw_striding', 'subsampling_factor': 8, 'subsampling_conv_channels': 256,
  'causal_downsampling': False, 'reduction': None, 'reduction_position': None,
  'reduction_factor': 1, 'ff_expansion_factor': 4,
  'self_attention_model': 'rel_pos', 'n_heads': 8, 'att_context_size': [-1, -1],
  'att_chunk_context_size': [[70], [1, 2, 7, 13], [0, 1, 2, 3, 4, 7, 13]], 'att_context_style': 'chunked_limited_with_rc',
  'xscaling': True, 'untie_biases': True, 'pos_emb_max_len': 5000, 'conv_kernel_size': 9, 'conv_norm_type': 'batch_norm',
  'conv_context_size': None, 'conv_context_style': 'dcc', 'dropout': 0.1, 'dropout_pre_encoder': 0.1, 'dropout_emb': 0.0,
  'dropout_att': 0.1, 'stochastic_depth_drop_prob': 0.0, 'stochastic_depth_mode': 'linear', 'stochastic_depth_start_layer': 1},
'decoder': {'_target_': 'nemo.collections.asr.modules.RNNTDecoder',
  'normalization_mode': None, 'random_state_sampling': False, 'blank_as_pad': True,
  'prednet': {'pred_hidden': 640, 'pred_rnn_layers': 2, 't_max': None, 'dropout': 0.2},
  'vocab_size': 1024},
'joint': {
    '_target_': 'nemo.collections.asr.modules.RNNTJoint', 'log_softmax': None,
    'preserve_memory': False, 'fuse_loss_wer': False,
    'fused_batch_size': -1,
    'jointnet': {'joint_hidden': 640, 'activation': 'relu', 'dropout': 0.2,
       'encoder_hidden': 1024, 'pred_hidden': 640}, 'num_classes': 1024,
       'vocabulary': ['<unk>', '▁t', '▁th', ... ... '&', '*', '/', '£', '+', '€', '_', '^', '¥']},
'decoding': {'strategy': 'greedy_batch', 'greedy': {'max_symbols': 10, 'use_cuda_graph_decoder': False},
  'beam': {'beam_size': 2, 'return_best_hypothesis': False, 'score_norm': True, 'tsd_max_sym_exp': 50, 'alsd_max_target_len': 2.0}},
'loss': {'loss_name': 'default', 'offline_loss_weight': 0.3, 'streaming_loss_weight': 0.7},
'optim': {'name': 'adamw', 'lr': 0.0001, 'betas': [0.9, 0.98], 'weight_decay': 0.001, 'sched': {'name': 'CosineAnnealing', 'warmup_steps': 3000, 'warmup_ratio': None, 'min_lr': 5e-06}},
'labels': ['<unk>', '▁t', '▁th', '▁a', 'in', '▁the', ... .... '"', '&', '*', '/', '£', '+', '€', '_', '^', '¥'],
'target': 'nemo.collections.asr.models.rnnt_bpe_models.EncDecRNNTBPEModel', 'nemo_version': '2.7.0rc0'}
```
