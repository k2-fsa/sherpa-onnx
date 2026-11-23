#!/usr/bin/env python3
"""Build tiny ONNX control-flow fixtures and run the actual C++ generator.

Requires numpy and onnx. No checkpoint/network needed. --tokenizer optionally
uses a real local Qwen tokenizer instead of the self-contained byte vocabulary.
Fixtures test runtime contracts, not recognition quality.
"""
import argparse
import os
from pathlib import Path
import shutil
import subprocess

import numpy as np
from onnx import TensorProto as T, checker, helper as h, numpy_helper, save
import json


def value(name, dtype, shape):
    return h.make_tensor_value_info(name, dtype, shape)


def model(path, nodes, inputs, outputs, constants):
    graph = h.make_graph(nodes, 'qwen_control_flow', inputs, outputs,
        [numpy_helper.from_array(np.asarray(v), k) for k, v in constants.items()])
    m = h.make_model(graph, opset_imports=[h.make_opsetid('', 17)], ir_version=9)
    checker.check_model(m)
    save(m, path)


def fixtures(directory, tokenizer):
    directory.mkdir(parents=True, exist_ok=False)
    tok = directory / 'tokenizer'
    if tokenizer:
        tok.mkdir()
        for name in ['vocab.json', 'merges.txt', 'tokenizer_config.json']:
            shutil.copyfile(tokenizer / name, tok / name)
    else:
        tok.mkdir()
        bs = list(range(33, 127)) + list(range(161, 173)) + list(range(174, 256))
        cs = bs.copy()
        for b in range(256):
            if b not in bs:
                bs.append(b)
                cs.append(256 + len(cs) - 188)
        # GPT-2 byte-to-Unicode mapping; original visible alphabet has 188 bytes.
        vocab = {chr(c): i for i, c in enumerate(cs)}
        vocab.update({'<|im_start|>':151644, '<|im_end|>':151645,
            '<|audio_start|>':151669, '<|audio_end|>':151670,
            '<|audio_pad|>':151676, '<asr_text>':151704})
        (tok / 'vocab.json').write_text(json.dumps(vocab), encoding='utf-8')
        (tok / 'merges.txt').write_text('#version: 0.2\n', encoding='utf-8')
        (tok / 'tokenizer_config.json').write_text('{}', encoding='utf-8')
    for name, fill in [('conv', 1), ('silence', 0)]:
        model(directory / (name+'.onnx'), [h.make_node('Identity',['data'],['out'])],
            [value('mel',T.FLOAT,[1,'frames',128])], [value('out',T.FLOAT,[1,1,4])],
            {'data':np.full((1,1,4),fill,dtype=np.float32)})
    for name, shape in [('malformed', (1,4)), ('empty', (1,0,4))]:
        model(directory / (name+'.onnx'), [h.make_node('Identity',['data'],['out'])],
            [value('mel',T.FLOAT,[1,'frames',128])], [value('out',T.FLOAT,list(shape))],
            {'data':np.ones(shape,dtype=np.float32)})
    model(directory/'encoder.onnx', [h.make_node('Identity',['features'],['out'])],
        [value('features',T.FLOAT,[1,'audio',4]), value('mask',T.BOOL,[1,'audio'])],
        [value('out',T.FLOAT,[1,'audio',4])], {})
    inputs = [value('input_ids',T.INT64,[1,'seq']),
        value('audio_features',T.FLOAT,[1,'audio',4]),
        value('attention_mask',T.INT64,[1,'seq']), value('cache_position',T.INT64,['seq']),
        value('past_key',T.FLOAT,[1,512,1,1]), value('past_value',T.FLOAT,[1,512,1,1])]
    outputs = [value('logits',T.FLOAT,[1,'seq',151705]),
        value('key_delta',T.FLOAT,[1,'seq',1,1]), value('value_delta',T.FLOAT,[1,'seq',1,1])]
    for mode in ['eos','observe','repeat']:
        constants = {k:np.array(v,dtype=np.int64) for k,v in {
            'zero':0,'one':1,'two':2,'last':-1,'B':33,'C':34,'D':35,'A':32,
            'EOS':151645,'vocab_size':151705,'vocab_tail':[151705],
            'axes':[0,2,3],'row_axes':[0,1],'float_axes':[0],
            'kv_tail':[1,1]}.items()}
        constants.update(high=np.array(10,dtype=np.float32),
            low=np.array(-100,dtype=np.float32), fallback=np.array(9,dtype=np.float32))
        nodes = [h.make_node('Shape',['input_ids'],['shape']),
            h.make_node('Gather',['input_ids','last'],['last_id'],axis=1),
            h.make_node('Gather',['cache_position','zero'],['start']),
            h.make_node('Gather',['shape','one'],['seq']),
            h.make_node('Add',['start','seq'],['end']),
            h.make_node('Range',['start','end','one'],['expected_pos']),
            h.make_node('Equal',['cache_position','expected_pos'],['pos_ok']),
            h.make_node('Cast',['pos_ok'],['pos_int'],to=T.INT64),
            h.make_node('ReduceMin',['pos_int'],['all_pos_ok'],keepdims=0),
            h.make_node('Add',['start','one'],['start1']),
            h.make_node('Mul',['start','start1'],['sum2']),
            h.make_node('Div',['sum2','two'],['sum_int']),
            h.make_node('Cast',['sum_int'],['expected_sum'],to=T.FLOAT),
            h.make_node('ReduceSum',['past_key'],['actual_sum'],keepdims=0),
            h.make_node('Equal',['actual_sum','expected_sum'],['kv_ok']),
            h.make_node('Cast',['all_pos_ok'],['positions_ok'],to=T.BOOL),
            h.make_node('And',['positions_ok','kv_ok'],['valid']),
            h.make_node('Equal',['last_id','B'],['is_B']),
            h.make_node('Equal',['last_id','C'],['is_C']),
            h.make_node('Where',['is_C','D','EOS'],['after_C']),
            h.make_node('Where',['is_B','C','after_C'],['following']),
            h.make_node('Where',['valid','following','A'],['observed']),
            h.make_node('Identity',[{'eos':'EOS','repeat':'A','observe':'observed'}[mode]],['chosen']),
            h.make_node('Range',['zero','vocab_size','one'],['vocab']),
            h.make_node('Equal',['vocab','chosen'],['best']),
            h.make_node('Equal',['vocab','A'],['fallback_token']),
            h.make_node('Where',['fallback_token','fallback','low'],['fallback_row']),
            h.make_node('Where',['best','high','fallback_row'],['row']),
            h.make_node('Concat',['shape','vocab_tail'],['logits_shape'],axis=0),
            h.make_node('Expand',['row','logits_shape'],['logits']),
            h.make_node('Add',['cache_position','one'],['kv_pos']),
            h.make_node('Cast',['kv_pos'],['kv_float'],to=T.FLOAT),
            h.make_node('Unsqueeze',['kv_float','axes'],['key_delta']),
            h.make_node('Identity',['key_delta'],['value_delta'])]
        model(directory/(mode+'.onnx'),nodes,inputs,outputs,constants)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--binary',type=Path,required=True)
    p.add_argument('--tokenizer',type=Path)
    args=p.parse_args()
    fixtures(args.output.resolve(), args.tokenizer)
    env=dict(os.environ,QWEN3_TEST_MODEL_DIR=str(args.output.resolve()))
    subprocess.run([str(args.binary.resolve(strict=True)),
        '--gtest_filter=QwenContinuationTest.*'],env=env,check=True,timeout=180)


if __name__=='__main__':
    main()
