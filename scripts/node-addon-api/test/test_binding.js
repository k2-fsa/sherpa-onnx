const sherpa_onnx = require('../lib/binding.js');
console.log(sherpa_onnx)

const config = {
  'featConfig': {
    'sampleRate': 16000,
    'featureDim': 80,
  }
};
const onlineRecognizer = sherpa_onnx.createOnlineRecognizer(config)

console.log('Tests passed- everything looks OK!');
