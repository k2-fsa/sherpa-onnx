const sherpa_onnx = require('../lib/binding.js');
console.log(sherpa_onnx)

const config = {
  'featConfig': {
    'sampleRate': 16000,
    'featureDim': 80,
  },
  'modelConfig': {
    'transducer': {
      'encoder': './encoder.onnx',
      'decoder': './decoder.onnx',
      'joiner': './joiner.onnx',
    },
    'tokens': './tokens.txt',
    'numThreads': 3,
    'provider': 'cpu',
    'debug': 10,
    'modelType': 'zipformer2',
  }
};
const onlineRecognizer = sherpa_onnx.createOnlineRecognizer(config)

console.log('Tests passed- everything looks OK!');
