# Notes

## asr_model
```
EncDecRNNTBPEModel(
  (preprocessor): AudioToMelSpectrogramPreprocessor(
    (featurizer): FilterbankFeatures()
  )
  (encoder): ConformerEncoder(
    (pre_encode): ConvSubsampling(
      (out): Linear(in_features=4096, out_features=1024, bias=True)
      (conv): MaskedConvSequential(
        (0): Conv2d(1, 256, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
        (1): ReLU(inplace=True)
        (2): Conv2d(256, 256, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), groups=256)
        (3): Conv2d(256, 256, kernel_size=(1, 1), stride=(1, 1))
        (4): ReLU(inplace=True)
        (5): Conv2d(256, 256, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), groups=256)
        (6): Conv2d(256, 256, kernel_size=(1, 1), stride=(1, 1))
        (7): ReLU(inplace=True)
      )
    )
    (pos_enc): RelPositionalEncoding(
      (dropout): Dropout(p=0.1, inplace=False)
    )
    (layers): ModuleList(
      (0-23): 24 x ConformerLayer(
        (norm_feed_forward1): LayerNorm((1024,), eps=1e-05, elementwise_affine=True)
        (feed_forward1): ConformerFeedForward(
          (linear1): Linear(in_features=1024, out_features=4096, bias=True)
          (activation): Swish()
          (dropout): Dropout(p=0.1, inplace=False)
          (linear2): Linear(in_features=4096, out_features=1024, bias=True)
        )
        (norm_conv): LayerNorm((1024,), eps=1e-05, elementwise_affine=True)
        (conv): ConformerConvolution(
          (pointwise_conv1): Conv1d(1024, 2048, kernel_size=(1,), stride=(1,))
          (depthwise_conv): CausalConv1D(1024, 1024, kernel_size=(9,), stride=(1,), groups=1024)
          (batch_norm): BatchNorm1d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (activation): Swish()
          (pointwise_conv2): Conv1d(1024, 1024, kernel_size=(1,), stride=(1,))
        )
        (norm_self_att): LayerNorm((1024,), eps=1e-05, elementwise_affine=True)
        (self_attn): RelPositionMultiHeadAttention(
          (linear_q): Linear(in_features=1024, out_features=1024, bias=True)
          (linear_k): Linear(in_features=1024, out_features=1024, bias=True)
          (linear_v): Linear(in_features=1024, out_features=1024, bias=True)
          (linear_out): Linear(in_features=1024, out_features=1024, bias=True)
          (dropout): Dropout(p=0.1, inplace=False)
          (linear_pos): Linear(in_features=1024, out_features=1024, bias=False)
        )
        (norm_feed_forward2): LayerNorm((1024,), eps=1e-05, elementwise_affine=True)
        (feed_forward2): ConformerFeedForward(
          (linear1): Linear(in_features=1024, out_features=4096, bias=True)
          (activation): Swish()
          (dropout): Dropout(p=0.1, inplace=False)
          (linear2): Linear(in_features=4096, out_features=1024, bias=True)
        )
        (dropout): Dropout(p=0.1, inplace=False)
        (norm_out): LayerNorm((1024,), eps=1e-05, elementwise_affine=True)
      )
    )
  )
  (decoder): RNNTDecoder(
    (prediction): ModuleDict(
      (embed): Embedding(1025, 640, padding_idx=1024)
      (dec_rnn): LSTMDropout(
        (lstm): LSTM(640, 640, num_layers=2, dropout=0.2)
        (dropout): Dropout(p=0.2, inplace=False)
      )
    )
  )
  (joint): RNNTJoint(
    (pred): Linear(in_features=640, out_features=640, bias=True)
    (enc): Linear(in_features=1024, out_features=640, bias=True)
    (joint_net): Sequential(
      (0): ReLU(inplace=True)
      (1): Dropout(p=0.2, inplace=False)
      (2): Linear(in_features=640, out_features=1025, bias=True)
    )
  )
  (loss): RNNTLoss(
    (_loss): RNNTLossNumba()
  )
  (spec_augmentation): SpectrogramAugmentation(
    (spec_augment): SpecAugment()
  )
  (wer): WER()
)
```

## asr_model.cfg
```
{'sample_rate': 16000, 'compute_eval_loss': False, 'log_prediction': True, 'rnnt_reduction': 'mean_volume', 'skip_nan_grad': False, 'model_defaults': {'enc_hidden': 1024, 'pred_hidden': 640, 'joint_hidden': 640}, 'train_ds': {'use_lhotse': True, 'skip_missing_manifest_entries': True, 'input_cfg': None, 'manifest_filepath': None, 'sample_rate': 16000, 'batch_size': 16, 'shuffle': True, 'num_workers': 8, 'pin_memory': True, 'max_duration': 40.0, 'min_duration': 0.1, 'text_field': 'answer', 'batch_duration': None, 'use_bucketing': True, 'max_tps': None, 'bucket_duration_bins': None, 'bucket_batch_size': None, 'num_buckets': None, 'bucket_buffer_size': None, 'shuffle_buffer_size': None, 'tarred_audio_filepaths': None, 'augmentor': None}, 'validation_ds': {}, 'tokenizer': {'dir': None, 'type': 'bpe', 'model_path': 'nemo:c9e35cde64e14bdc87cf70d543842217_tokenizer.model', 'vocab_path': 'nemo:28f042954ba747e99209b8ca5a223ba3_vocab.txt', 'spe_tokenizer_vocab': 'nemo:aa68b93b03344274b0c0e2a96333de24_tokenizer.vocab'}, 'preprocessor': {'_target_': 'nemo.collections.asr.modules.AudioToMelSpectrogramPreprocessor', 'sample_rate': 16000, 'normalize': 'per_feature', 'window_size': 0.025, 'window_stride': 0.01, 'window': 'hann', 'features': 128, 'n_fft': 512, 'frame_splicing': 1, 'dither': 1e-05, 'pad_to': 0, 'pad_value': 0.0}, 'spec_augment': {'_target_': 'nemo.collections.asr.modules.SpectrogramAugmentation', 'freq_masks': 2, 'time_masks': 10, 'freq_width': 27, 'time_width': 0.05}, 'encoder': {'_target_': 'nemo.collections.asr.modules.ConformerEncoder', 'feat_in': 128, 'feat_out': -1, 'n_layers': 24, 'd_model': 1024, 'subsampling': 'dw_striding', 'subsampling_factor': 8, 'subsampling_conv_channels': 256, 'causal_downsampling': False, 'reduction': None, 'reduction_position': None, 'reduction_factor': 1, 'ff_expansion_factor': 4, 'self_attention_model': 'rel_pos', 'n_heads': 8, 'att_context_size': [-1, -1], 'att_chunk_context_size': [[70], [1, 2, 7, 13], [0, 1, 2, 3, 4, 7, 13]], 'att_context_style': 'chunked_limited_with_rc', 'xscaling': True, 'untie_biases': True, 'pos_emb_max_len': 5000, 'conv_kernel_size': 9, 'conv_norm_type': 'batch_norm', 'conv_context_size': None, 'conv_context_style': 'dcc', 'dropout': 0.1, 'dropout_pre_encoder': 0.1, 'dropout_emb': 0.0, 'dropout_att': 0.1, 'stochastic_depth_drop_prob': 0.0, 'stochastic_depth_mode': 'linear', 'stochastic_depth_start_layer': 1}, 'decoder': {'_target_': 'nemo.collections.asr.modules.RNNTDecoder', 'normalization_mode': None, 'random_state_sampling': False, 'blank_as_pad': True, 'prednet': {'pred_hidden': 640, 'pred_rnn_layers': 2, 't_max': None, 'dropout': 0.2}, 'vocab_size': 1024}, 'joint': {'_target_': 'nemo.collections.asr.modules.RNNTJoint', 'log_softmax': None, 'preserve_memory': False, 'fuse_loss_wer': False, 'fused_batch_size': -1, 'jointnet': {'joint_hidden': 640, 'activation': 'relu', 'dropout': 0.2, 'encoder_hidden': 1024, 'pred_hidden': 640}, 'num_classes': 1024, 'vocabulary': ['<unk>', '▁t', '▁th', '▁a', 'in', '▁the', 're', '▁w', '▁o', '▁s', 'er', 'at', 'ou', 'nd', 'it', 'is', '▁h', '▁b', 'on', '▁c', 'ing', 'en', '▁to', '▁m', '▁f', '▁p', 'or', 'an', 'es', '▁of', '▁d', 'ed', 'll', '▁and', '▁I', '▁in', '▁l', 'ar', '▁y', '▁g', 'as', '▁you', 'om', '▁n', 'ic', 've', 'al', 'ion', 'us', '▁be', 'ow', 'le', '▁wh', '▁e', 'ot', 'ut', '▁it', '▁is', '▁we', '▁T', '▁re', 'et', '▁A', 'ent', '▁on', '▁ha', 'ay', '▁S', 'ct', '▁Th', 'ver', 'id', 'ig', 'im', 'ro', '▁for', 'ly', '▁he', 'ke', 'ld', 'se', 'st', 'ch', '▁st', 'all', 'ce', 'ur', 'ith', 'am', 'if', 'ir', '▁go', '▁u', '▁as', '▁was', 'ad', '▁W', '▁k', '▁an', 'ht', 'th', '▁r', '▁are', 'ere', '▁se', '▁do', '▁B', '▁so', '▁sh', '▁not', '▁li', 'od', '▁C', 'ust', 'ill', 'ight', 'ally', '▁And', 'ter', '▁or', '▁me', '▁M', 'ome', 'op', '▁at', 'il', '▁The', 'ould', '▁j', 'ant', '▁So', '▁H', 'ol', 'ain', '▁can', '▁de', '▁ne', 'ore', '▁con', '▁kn', 'ck', 'ul', '▁fr', '▁ab', 'ers', 'ess', 'ge', '▁pro', 'pe', 'ate', '▁su', '▁com', '▁but', '▁all', 'est', 'qu', '▁ex', '▁al', 'ra', '▁O', 'out', 'use', 'very', 'pp', '▁Y', '▁ch', 'ri', 'ist', '▁v', '▁lo', 'ment', 'art', '▁P', 'nt', 'ab', '▁one', '▁N', 'ive', '▁wor', 'ions', 'ort', '▁L', '▁by', 'ich', '▁my', 'ity', 'ok', '▁G', 'res', '▁up', 'un', 'um', 'ea', 'ind', 'and', 'ink', 'el', '▁D', 'em', '▁E', 'os', 'oug', '▁if', 'ca', '▁out', '▁int', 'ie', '▁F', '▁It', '▁his', 'ard', '▁had', '▁tr', 'her', 'our', 'ies', 'ake', '▁R', '▁We', '▁get', '▁don', '▁us', 'ak', '▁pl', 'ect', 'ure', 'ame', 'ast', '▁who', 'ack', '▁le', '▁sa', 'iv', 'ci', 'ide', '▁tim', '▁our', 'ound', 'ous', '▁co', '▁pe', 'ose', 'ud', '▁see', 'ough', '▁man', '▁qu', '▁You', 'so', 'ople', '▁Wh', 'ong', 'ap', 'ther', '▁J', 'are', 'ine', '▁say', '▁im', '▁But', 'ings', '▁has', '▁ag', 'ff', '▁her', 'itt', 'one', '▁en', '▁ar', '▁fe', 'ven', '▁any', '▁mo', 'reat', 'ag', '▁how', '▁cl', 'pt', '▁now', 'own', 'ber', '▁him', '▁act', 'hing', 'ice', '▁no', 'ans', 'iz', '▁fa', 'per', 'pl', '▁te', '▁ad', 'age', 'ree', '▁tw', 'ank', '▁He', 'ple', 'ite', 'ry', '▁U', 'ish', 'ire', 'ue', '▁In', '▁she', 'ble', 'cc', 'nder', '▁way', '▁pr', 'ear', '▁did', '▁po', 'eah', '▁un', 'omet', 'ence', 'ep', 'uch', '▁sp', 'ach', 'og', 'ance', 'able', 'iff', 'sel', '▁got', 'way', '▁gr', 'alk', '▁res', 'ated', 'irst', 'ick', 'ass', '▁two', '▁dis', 'ord', '▁pre', 'ount', 'ase', 'ip', 'ult', 'ical', 'orm', 'ary', 'ace', '▁spe', '▁Ch', '▁thr', '▁imp', 'int', '▁am', '▁off', 'act', 'ia', '▁ro', 'ress', '▁per', '▁fo', '▁br', '▁K', 'vel', '▁gu', '▁bo', 'ang', 'kay', 'ub', 'ign', '▁may', 'ving', 'ces', 'ens', 'cl', '▁lot', 'ru', 'ade', '▁bet', '▁bl', '▁let', 'fore', 'co', 'ild', 'ning', 'xt', 'ile', 'ark', 'self', '▁app', 'ory', 'du', '▁day', '▁St', 'ater', '▁use', 'ys', 'fter', '▁new', 'ious', 'ial', 'he', 'wn', 'ved', 'red', '▁fl', 'iss', 'ody', 'form', 'ian', 'tain', '▁bu', '▁V', '▁rec', 'ty', 'be', '▁sc', 'ors', 'vers', '▁put', 'ife', '▁If', 'we', 'te', 'ject', 'ath', 'ting', '▁rem', '▁acc', 'ull', 'ons', '▁ind', '▁ser', '▁ke', 'ates', 'ves', 'na', 'lic', '▁des', '▁its', 'ful', 'ents', 'erm', 'ac', 'ered', 'ise', '▁sy', 'urn', '▁em', 'oth', 'ual', 'ne', 'ward', 'ib', '▁try', '▁pos', 'nds', 'ft', 'get', 'ph', '▁ob', 'ady', 'igh', 'ood', '▁rel', '▁wr', 'ug', 'ears', 'ail', '▁Now', '▁bit', 'ng', '▁Oh', '▁hel', 'ange', '▁reg', '▁rep', '▁bel', '▁sm', 'ost', 'tern', 'gr', '▁own', '▁end', 'pect', 'ily', 'day', 'ied', 'ific', 'ower', '▁add', 'cess', 'ict', 'ible', '▁bas', '▁i', '▁op', 'cial', 'ular', '▁Be', 'ced', '▁too', 'ks', 'ew', 'mer', '▁ph', 'ob', '==', '▁la', '▁set', '▁min', '▁sub', '▁gen', 'atch', '..', '▁inv', '▁As', '▁nat', '▁sl', '▁num', 'av', 'ways', '▁God', 'stem', '▁ac', '▁att', '▁ev', '▁def', 'llow', '▁str', 'lect', 'ars', '▁cr', '▁Is', 'olog', 'les', 'oy', '▁ask', '▁inc', 'body', '▁ent', '▁pol', 'ness', 'ix', '▁why', 'onna', '▁ear', '▁tak', '▁Un', 'ited', 'mun', 'li', 'ute', 'ract', '▁dec', 'uro', '▁mak', '▁fin', 'ween', '▁No', 'arch', '▁bec', 'gan', 'old', 'cy', '▁big', '▁For', 'ren', 'als', 'und', '▁Al', '▁All', 'ss', 'ows', '▁mod', 'ock', '▁id', 'ism', 'cus', '▁gl', 'ably', '▁ass', '▁car', 'ata', 'ppen', 'led', '▁sim', '▁mon', 'ics', '▁giv', 'cept', '▁Mr', 'pan', '▁pub', '▁eff', '▁How', 'ps', 'vern', 'end', 'hip', 'iew', 'ope', '▁An', '▁She', '▁Com', 'ee', 'ures', 'ell', 'ouse', 'cond', 'king', 'oc', 'ues', 'ever', '▁To', 'clud', '▁ins', '▁exp', '▁old', '▁mem', '▁ref', '▁tra', '▁far', 'ave', 'rat', '▁sur', 'ruct', 'rib', 'duct', 'uff', '▁met', '▁sch', 'ince', '▁run', 'ense', '▁cle', '▁==', 'mon', 'ize', '▁ord', 'blem', 'tin', '▁Let', 'ner', 'ond', 'its', '▁cor', 'land', '▁cur', '▁Re', '▁bus', '▁uh', 'air', 'ote', 'ants', 'ason', 'ric', '▁el', '▁cer', 'nce', '▁fam', '▁cap', 'uck', 'ool', 'ried', '▁cou', '▁fun', '▁wom', '▁hum', '▁ty', '▁ap', 'ike', '▁few', 'oney', '▁inf', 'ont', 'ese', 'ook', 'gy', 'uth', 'ulat', 'ieve', 'ized', 'ross', '▁ple', '▁um', '▁val', '▁equ', '▁lea', '▁lar', 'ah', 'eral', '▁ed', 'ared', 'lish', 'arn', 'ds', 'esn', '▁iss', '▁ca', 'ted', 'ices', '▁wee', 'ash', '▁top', 'ten', 'up', 'ts', 'gin', 'con', 'ari', '▁opp', 'osed', '▁eas', '▁ext', 'gg', 'az', '▁Fr', 'ideo', 'izat', '▁men', '▁mom', '▁ret', 'tty', 'rist', '▁gra', 'alth', 'ef', '▁det', 'ax', '▁mat', 'chn', 'ern', 'peri', '▁bre', '▁Sh', 'sw', 'erat', '▁sit', 'ters', 'ale', 'man', '▁sol', 'ork', '▁adv', 'ety', '▁vis', '▁med', 'uc', 'less', '▁unt', 'gram', 'ets', 'ists', '▁ey', '▁col', 'imes', '▁law', '▁pri', 'sid', '▁On', '▁mot', 'ield', '▁Do', '▁At', 'ages', 'amp', '▁art', 'miss', '▁sk', 'alf', 'pr', 'ier', '▁beh', '▁Yes', 'ural', 'ime', '▁wa', 'oks', 'bers', 'ger', 'ient', 'ries', '...', '▁che', '▁Br', 'ird', '▁Ar', '▁war', 'inat', '▁My', 'ital', 'wh', 'med', '▁pur', 'ully', '▁One', '▁rat', 'ines', '▁Of', 'io', '▁loc', 'ret', 'ctor', '▁leg', 'stit', 'ined', 'ught', '▁dur', '▁es', 'vent', 'aj', '▁bro', '▁saw', '▁sec', 'ream', '▁pop', 'reen', '▁Ind', 'els', '▁yet', 'ired', '▁sw', 'tro', 'oup', 'most', 'pean', 'eds', 'ush', 'oh', '▁Se', '▁tea', 'ann', 'ilit', 'err', 'pend', 'ton', 'ased', '▁aff', '▁mor', '▁dra', 'put', '▁dr', 'ins', 'uat', 'nect', 'cri', 'outh', '▁ra', '▁pay', 'ms', '▁av', 'bs', 'ling', '▁De', '▁Or', 'ove', '▁Can', '▁eng', 'ames', 'ided', '▁Go', 'mitt', 'ode', '▁cre', 'par', 'ides', 'pos', '▁fav', '▁air', '▁New', '▁bad', '▁six', 'vat', '▁pat', 'not', '▁di', 'rop', 'ral', 'orn', '▁par', 'cing', '▁aw', 'orts', 'ox', '▁yes', 'cuss', 'eng', 'ives', 'erms', '▁job', 'mand', 'ying', '▁occ', 'aps', 'ases', '▁Not', 'rent', 'ency', 'att', 'ised', 'vice', '▁Eng', '▁est', 'oked', '▁Q', 'iron', 'idd', 'me', 'unch', 'ane', '▁z', 'br', 'arts', '▁fat', 'ery', 'anks', '▁jo', '▁mar', 'aw', 'ott', 'ards', '▁oh', 'ians', '▁sci', 'row', 'unt', 'ury', '▁abs', 'ergy', '▁Z', 'ump', '▁Am', 'ened', 'angu', '▁Pro', 'icat', 'itch', '▁dri', 'iat', '▁', 'e', 't', 'o', 'a', 'n', 'i', 's', 'r', 'h', 'l', 'd', 'u', 'c', 'm', 'y', 'g', 'w', 'f', 'p', ',', '.', 'b', 'v', 'k', "'", 'I', 'T', 'A', 'S', 'x', 'W', 'j', 'C', 'B', 'M', '?', 'H', 'O', '0', 'P', 'q', 'Y', 'N', 'L', 'D', '1', 'E', 'G', 'z', 'F', 'R', '-', '2', 'J', 'U', '9', 'K', '5', '3', 'V', '=', '4', '8', '6', '7', '!', '%', ':', 'Q', 'Z', '$', 'X', '"', '&', '*', '/', '£', '+', '€', '_', '^', '¥']}, 'decoding': {'strategy': 'greedy_batch', 'greedy': {'max_symbols': 10, 'use_cuda_graph_decoder': False}, 'beam': {'beam_size': 2, 'return_best_hypothesis': False, 'score_norm': True, 'tsd_max_sym_exp': 50, 'alsd_max_target_len': 2.0}}, 'loss': {'loss_name': 'default', 'offline_loss_weight': 0.3, 'streaming_loss_weight': 0.7}, 'optim': {'name': 'adamw', 'lr': 0.0001, 'betas': [0.9, 0.98], 'weight_decay': 0.001, 'sched': {'name': 'CosineAnnealing', 'warmup_steps': 3000, 'warmup_ratio': None, 'min_lr': 5e-06}}, 'labels': ['<unk>', '▁t', '▁th', '▁a', 'in', '▁the', 're', '▁w', '▁o', '▁s', 'er', 'at', 'ou', 'nd', 'it', 'is', '▁h', '▁b', 'on', '▁c', 'ing', 'en', '▁to', '▁m', '▁f', '▁p', 'or', 'an', 'es', '▁of', '▁d', 'ed', 'll', '▁and', '▁I', '▁in', '▁l', 'ar', '▁y', '▁g', 'as', '▁you', 'om', '▁n', 'ic', 've', 'al', 'ion', 'us', '▁be', 'ow', 'le', '▁wh', '▁e', 'ot', 'ut', '▁it', '▁is', '▁we', '▁T', '▁re', 'et', '▁A', 'ent', '▁on', '▁ha', 'ay', '▁S', 'ct', '▁Th', 'ver', 'id', 'ig', 'im', 'ro', '▁for', 'ly', '▁he', 'ke', 'ld', 'se', 'st', 'ch', '▁st', 'all', 'ce', 'ur', 'ith', 'am', 'if', 'ir', '▁go', '▁u', '▁as', '▁was', 'ad', '▁W', '▁k', '▁an', 'ht', 'th', '▁r', '▁are', 'ere', '▁se', '▁do', '▁B', '▁so', '▁sh', '▁not', '▁li', 'od', '▁C', 'ust', 'ill', 'ight', 'ally', '▁And', 'ter', '▁or', '▁me', '▁M', 'ome', 'op', '▁at', 'il', '▁The', 'ould', '▁j', 'ant', '▁So', '▁H', 'ol', 'ain', '▁can', '▁de', '▁ne', 'ore', '▁con', '▁kn', 'ck', 'ul', '▁fr', '▁ab', 'ers', 'ess', 'ge', '▁pro', 'pe', 'ate', '▁su', '▁com', '▁but', '▁all', 'est', 'qu', '▁ex', '▁al', 'ra', '▁O', 'out', 'use', 'very', 'pp', '▁Y', '▁ch', 'ri', 'ist', '▁v', '▁lo', 'ment', 'art', '▁P', 'nt', 'ab', '▁one', '▁N', 'ive', '▁wor', 'ions', 'ort', '▁L', '▁by', 'ich', '▁my', 'ity', 'ok', '▁G', 'res', '▁up', 'un', 'um', 'ea', 'ind', 'and', 'ink', 'el', '▁D', 'em', '▁E', 'os', 'oug', '▁if', 'ca', '▁out', '▁int', 'ie', '▁F', '▁It', '▁his', 'ard', '▁had', '▁tr', 'her', 'our', 'ies', 'ake', '▁R', '▁We', '▁get', '▁don', '▁us', 'ak', '▁pl', 'ect', 'ure', 'ame', 'ast', '▁who', 'ack', '▁le', '▁sa', 'iv', 'ci', 'ide', '▁tim', '▁our', 'ound', 'ous', '▁co', '▁pe', 'ose', 'ud', '▁see', 'ough', '▁man', '▁qu', '▁You', 'so', 'ople', '▁Wh', 'ong', 'ap', 'ther', '▁J', 'are', 'ine', '▁say', '▁im', '▁But', 'ings', '▁has', '▁ag', 'ff', '▁her', 'itt', 'one', '▁en', '▁ar', '▁fe', 'ven', '▁any', '▁mo', 'reat', 'ag', '▁how', '▁cl', 'pt', '▁now', 'own', 'ber', '▁him', '▁act', 'hing', 'ice', '▁no', 'ans', 'iz', '▁fa', 'per', 'pl', '▁te', '▁ad', 'age', 'ree', '▁tw', 'ank', '▁He', 'ple', 'ite', 'ry', '▁U', 'ish', 'ire', 'ue', '▁In', '▁she', 'ble', 'cc', 'nder', '▁way', '▁pr', 'ear', '▁did', '▁po', 'eah', '▁un', 'omet', 'ence', 'ep', 'uch', '▁sp', 'ach', 'og', 'ance', 'able', 'iff', 'sel', '▁got', 'way', '▁gr', 'alk', '▁res', 'ated', 'irst', 'ick', 'ass', '▁two', '▁dis', 'ord', '▁pre', 'ount', 'ase', 'ip', 'ult', 'ical', 'orm', 'ary', 'ace', '▁spe', '▁Ch', '▁thr', '▁imp', 'int', '▁am', '▁off', 'act', 'ia', '▁ro', 'ress', '▁per', '▁fo', '▁br', '▁K', 'vel', '▁gu', '▁bo', 'ang', 'kay', 'ub', 'ign', '▁may', 'ving', 'ces', 'ens', 'cl', '▁lot', 'ru', 'ade', '▁bet', '▁bl', '▁let', 'fore', 'co', 'ild', 'ning', 'xt', 'ile', 'ark', 'self', '▁app', 'ory', 'du', '▁day', '▁St', 'ater', '▁use', 'ys', 'fter', '▁new', 'ious', 'ial', 'he', 'wn', 'ved', 'red', '▁fl', 'iss', 'ody', 'form', 'ian', 'tain', '▁bu', '▁V', '▁rec', 'ty', 'be', '▁sc', 'ors', 'vers', '▁put', 'ife', '▁If', 'we', 'te', 'ject', 'ath', 'ting', '▁rem', '▁acc', 'ull', 'ons', '▁ind', '▁ser', '▁ke', 'ates', 'ves', 'na', 'lic', '▁des', '▁its', 'ful', 'ents', 'erm', 'ac', 'ered', 'ise', '▁sy', 'urn', '▁em', 'oth', 'ual', 'ne', 'ward', 'ib', '▁try', '▁pos', 'nds', 'ft', 'get', 'ph', '▁ob', 'ady', 'igh', 'ood', '▁rel', '▁wr', 'ug', 'ears', 'ail', '▁Now', '▁bit', 'ng', '▁Oh', '▁hel', 'ange', '▁reg', '▁rep', '▁bel', '▁sm', 'ost', 'tern', 'gr', '▁own', '▁end', 'pect', 'ily', 'day', 'ied', 'ific', 'ower', '▁add', 'cess', 'ict', 'ible', '▁bas', '▁i', '▁op', 'cial', 'ular', '▁Be', 'ced', '▁too', 'ks', 'ew', 'mer', '▁ph', 'ob', '==', '▁la', '▁set', '▁min', '▁sub', '▁gen', 'atch', '..', '▁inv', '▁As', '▁nat', '▁sl', '▁num', 'av', 'ways', '▁God', 'stem', '▁ac', '▁att', '▁ev', '▁def', 'llow', '▁str', 'lect', 'ars', '▁cr', '▁Is', 'olog', 'les', 'oy', '▁ask', '▁inc', 'body', '▁ent', '▁pol', 'ness', 'ix', '▁why', 'onna', '▁ear', '▁tak', '▁Un', 'ited', 'mun', 'li', 'ute', 'ract', '▁dec', 'uro', '▁mak', '▁fin', 'ween', '▁No', 'arch', '▁bec', 'gan', 'old', 'cy', '▁big', '▁For', 'ren', 'als', 'und', '▁Al', '▁All', 'ss', 'ows', '▁mod', 'ock', '▁id', 'ism', 'cus', '▁gl', 'ably', '▁ass', '▁car', 'ata', 'ppen', 'led', '▁sim', '▁mon', 'ics', '▁giv', 'cept', '▁Mr', 'pan', '▁pub', '▁eff', '▁How', 'ps', 'vern', 'end', 'hip', 'iew', 'ope', '▁An', '▁She', '▁Com', 'ee', 'ures', 'ell', 'ouse', 'cond', 'king', 'oc', 'ues', 'ever', '▁To', 'clud', '▁ins', '▁exp', '▁old', '▁mem', '▁ref', '▁tra', '▁far', 'ave', 'rat', '▁sur', 'ruct', 'rib', 'duct', 'uff', '▁met', '▁sch', 'ince', '▁run', 'ense', '▁cle', '▁==', 'mon', 'ize', '▁ord', 'blem', 'tin', '▁Let', 'ner', 'ond', 'its', '▁cor', 'land', '▁cur', '▁Re', '▁bus', '▁uh', 'air', 'ote', 'ants', 'ason', 'ric', '▁el', '▁cer', 'nce', '▁fam', '▁cap', 'uck', 'ool', 'ried', '▁cou', '▁fun', '▁wom', '▁hum', '▁ty', '▁ap', 'ike', '▁few', 'oney', '▁inf', 'ont', 'ese', 'ook', 'gy', 'uth', 'ulat', 'ieve', 'ized', 'ross', '▁ple', '▁um', '▁val', '▁equ', '▁lea', '▁lar', 'ah', 'eral', '▁ed', 'ared', 'lish', 'arn', 'ds', 'esn', '▁iss', '▁ca', 'ted', 'ices', '▁wee', 'ash', '▁top', 'ten', 'up', 'ts', 'gin', 'con', 'ari', '▁opp', 'osed', '▁eas', '▁ext', 'gg', 'az', '▁Fr', 'ideo', 'izat', '▁men', '▁mom', '▁ret', 'tty', 'rist', '▁gra', 'alth', 'ef', '▁det', 'ax', '▁mat', 'chn', 'ern', 'peri', '▁bre', '▁Sh', 'sw', 'erat', '▁sit', 'ters', 'ale', 'man', '▁sol', 'ork', '▁adv', 'ety', '▁vis', '▁med', 'uc', 'less', '▁unt', 'gram', 'ets', 'ists', '▁ey', '▁col', 'imes', '▁law', '▁pri', 'sid', '▁On', '▁mot', 'ield', '▁Do', '▁At', 'ages', 'amp', '▁art', 'miss', '▁sk', 'alf', 'pr', 'ier', '▁beh', '▁Yes', 'ural', 'ime', '▁wa', 'oks', 'bers', 'ger', 'ient', 'ries', '...', '▁che', '▁Br', 'ird', '▁Ar', '▁war', 'inat', '▁My', 'ital', 'wh', 'med', '▁pur', 'ully', '▁One', '▁rat', 'ines', '▁Of', 'io', '▁loc', 'ret', 'ctor', '▁leg', 'stit', 'ined', 'ught', '▁dur', '▁es', 'vent', 'aj', '▁bro', '▁saw', '▁sec', 'ream', '▁pop', 'reen', '▁Ind', 'els', '▁yet', 'ired', '▁sw', 'tro', 'oup', 'most', 'pean', 'eds', 'ush', 'oh', '▁Se', '▁tea', 'ann', 'ilit', 'err', 'pend', 'ton', 'ased', '▁aff', '▁mor', '▁dra', 'put', '▁dr', 'ins', 'uat', 'nect', 'cri', 'outh', '▁ra', '▁pay', 'ms', '▁av', 'bs', 'ling', '▁De', '▁Or', 'ove', '▁Can', '▁eng', 'ames', 'ided', '▁Go', 'mitt', 'ode', '▁cre', 'par', 'ides', 'pos', '▁fav', '▁air', '▁New', '▁bad', '▁six', 'vat', '▁pat', 'not', '▁di', 'rop', 'ral', 'orn', '▁par', 'cing', '▁aw', 'orts', 'ox', '▁yes', 'cuss', 'eng', 'ives', 'erms', '▁job', 'mand', 'ying', '▁occ', 'aps', 'ases', '▁Not', 'rent', 'ency', 'att', 'ised', 'vice', '▁Eng', '▁est', 'oked', '▁Q', 'iron', 'idd', 'me', 'unch', 'ane', '▁z', 'br', 'arts', '▁fat', 'ery', 'anks', '▁jo', '▁mar', 'aw', 'ott', 'ards', '▁oh', 'ians', '▁sci', 'row', 'unt', 'ury', '▁abs', 'ergy', '▁Z', 'ump', '▁Am', 'ened', 'angu', '▁Pro', 'icat', 'itch', '▁dri', 'iat', '▁', 'e', 't', 'o', 'a', 'n', 'i', 's', 'r', 'h', 'l', 'd', 'u', 'c', 'm', 'y', 'g', 'w', 'f', 'p', ',', '.', 'b', 'v', 'k', "'", 'I', 'T', 'A', 'S', 'x', 'W', 'j', 'C', 'B', 'M', '?', 'H', 'O', '0', 'P',
 'q', 'Y', 'N', 'L', 'D', '1', 'E', 'G', 'z', 'F', 'R', '-', '2', 'J', 'U', '9', 'K', '5', '3', 'V', '=', '4', '8', '6', '7', '!', '%', ':', 'Q', 'Z', '$', 'X', '"', '&', '*', '/', '£', '+', '€', '_', '^', '¥'], 'target': 'nemo.collections.asr.models.rnnt_bpe_models.EncDecRNNTBPEModel', 'nemo_version': '2.7.0rc0'}
```


## onnx
```
NodeArg(name='target_length', type='tensor(int32)', shape=['target_length_dynamic_axes_1'])
NodeArg(name='states.1', type='tensor(float)', shape=[2, 'states.1_dim_1', 640])
NodeArg(name='onnx::Slice_3', type='tensor(float)', shape=[2, 1, 640])
==========decoderOutput==========
NodeArg(name='outputs', type='tensor(float)', shape=['targets_dynamic_axes_1', 640, 'targets_dynamic_axes_2'])
NodeArg(name='prednet_lengths', type='tensor(int32)', shape=['target_length_dynamic_axes_1'])
NodeArg(name='states', type='tensor(float)', shape=[2, 'states_dynamic_axes_1', 640])
NodeArg(name='162', type='tensor(float)', shape=[2, 'Concat162_dim_1', 640])
==========joiner Input==========
NodeArg(name='encoder_outputs', type='tensor(float)', shape=['encoder_outputs_dynamic_axes_1', 1024, 'encoder_outputs_dynamic_axes_2'])
NodeArg(name='decoder_outputs', type='tensor(float)', shape=['decoder_outputs_dynamic_axes_1', 640, 'decoder_outputs_dynamic_axes_2'])
==========joinerOutput==========
NodeArg(name='outputs', type='tensor(float)', shape=['Addoutputs_dim_0', 'Addoutputs_dim_1', 'Addoutputs_dim_2', 1025])
(150960,)
features.shape (942, 128)
[218, 32, 961, 34, 220, 966, 943, 7, 302, 22, 243, 56, 271, 23, 137, 961, 461, 948, 10, 404, 172, 950, 944, 942, 416, 961, 1, 86, 385, 3, 329, 264, 755, 28, 56, 57, 659, 411, 76, 941, 162, 110, 78, 5, 619, 25, 180, 158, 14]
2086-149220-0033.wav
Well, I don't wish to see it any more, observed Phoebe, turning away her eyes it is certainly very like the old portrait
RTF: 0.10016013783756614
+ echo ---fp32----
---fp32----
+ python3 ./test_onnx.py --encoder ./encoder.int8.onnx --decoder ./decoder.onnx --joiner ./joiner.onnx --tokens ./tokens.txt --wav 2086-149220-0033.wav
{'encoder': './encoder.int8.onnx', 'decoder': './decoder.onnx', 'joiner': './joiner.onnx', 'tokens': './tokens.txt', 'wav': '2086-149220-0033.wav'}
{'vocab_size': '1024', 'comment': 'This model contains only the non-streaming part', 'model_author': 'NeMo', 'subsampling_factor': '8', 'url': 'https://huggingface.co/nvidia/parakeet-unified-en-0.6b', 'pred_hidden': '640', 'normalize_type': 'per_feature', 'pred_rnn_layers': '2', 'model_type': 'EncDecRNNTBPEModel', 'feat_dim': '128', 'version': '2'}
==========encoder Input==========
NodeArg(name='audio_signal', type='tensor(float)', shape=['audio_signal_dynamic_axes_1', 128, 'audio_signal_dynamic_axes_2'])
NodeArg(name='length', type='tensor(int64)', shape=['length_dynamic_axes_1'])
==========encoderOutput==========
NodeArg(name='outputs', type='tensor(float)', shape=['Transposeoutputs_dim_0', 1024, 'Transposeoutputs_dim_2'])
NodeArg(name='encoded_lengths', type='tensor(int64)', shape=['length_dynamic_axes_1'])
==========decoder Input==========
NodeArg(name='targets', type='tensor(int32)', shape=['targets_dynamic_axes_1', 'targets_dynamic_axes_2'])
NodeArg(name='target_length', type='tensor(int32)', shape=['target_length_dynamic_axes_1'])
NodeArg(name='states.1', type='tensor(float)', shape=[2, 'states.1_dim_1', 640])
NodeArg(name='onnx::Slice_3', type='tensor(float)', shape=[2, 1, 640])
==========decoderOutput==========
NodeArg(name='outputs', type='tensor(float)', shape=['targets_dynamic_axes_1', 640, 'targets_dynamic_axes_2'])
NodeArg(name='prednet_lengths', type='tensor(int32)', shape=['target_length_dynamic_axes_1'])
NodeArg(name='states', type='tensor(float)', shape=[2, 'states_dynamic_axes_1', 640])
NodeArg(name='162', type='tensor(float)', shape=[2, 'Concat162_dim_1', 640])
==========joiner Input==========
NodeArg(name='encoder_outputs', type='tensor(float)', shape=['encoder_outputs_dynamic_axes_1', 1024, 'encoder_outputs_dynamic_axes_2'])
NodeArg(name='decoder_outputs', type='tensor(float)', shape=['decoder_outputs_dynamic_axes_1', 640, 'decoder_outputs_dynamic_axes_2'])
==========joinerOutput==========
NodeArg(name='outputs', type='tensor(float)', shape=['Addoutputs_dim_0', 'Addoutputs_dim_1', 'Addoutputs_dim_2', 1025])
(150960,)
features.shape (942, 128)
[218, 32, 961, 34, 220, 966, 943, 7, 302, 22, 243, 56, 271, 23, 137, 961, 461, 948, 10, 404, 172, 950, 944, 942, 416, 961, 1, 86, 385, 3, 329, 264, 755, 28, 56, 57, 659, 411, 76, 941, 162, 110, 78, 5, 619, 25, 180, 158, 14]
2086-149220-0033.wav
Well, I don't wish to see it any more, observed Phoebe, turning away her eyes it is certainly very like the old portrait
RTF: 0.12250990228294126
-rw-r--r--  1 runner  staff   6.9M Apr 27 06:33 decoder.int8.onnx
-rw-r--r--  1 runner  staff    28M Apr 27 06:31 decoder.onnx
-rw-r--r--  1 runner  staff   624M Apr 27 06:33 encoder.int8.onnx
-rw-r--r--  1 runner  staff    40M Apr 27 06:34 encoder.onnx
-rw-r--r--  1 runner  staff   1.7M Apr 27 06:33 joiner.int8.onnx
-rw-r--r--  1 runner  staff   6.6M Apr 27 06:31 joiner.onnx
-rw-------  1 runner  staff   2.3G Apr 27 06:34 encoder.weights
```
