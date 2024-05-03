// Copyright (c)  2024  Xiaomi Corporation
const sherpa_onnx = require('../lib/sherpa-onnx.js');

const config = {
  'featConfig': {
    'sampleRate': 16000,
    'featureDim': 80,
  },
  'modelConfig': {
    'transducer': {
      'encoder':
          './sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20/encoder-epoch-99-avg-1.int8.onnx',
      'decoder':
          './sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20/decoder-epoch-99-avg-1.onnx',
      'joiner':
          './sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20/joiner-epoch-99-avg-1.int8.onnx',
    },
    'tokens':
        './sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20/tokens.txt',
    'numThreads': 1,
    'provider': 'cpu',
    'debug': 1,
    'modelType': 'zipformer',
  }
};
const onlineRecognizer = new sherpa_onnx.OnlineRecognizer(config)
