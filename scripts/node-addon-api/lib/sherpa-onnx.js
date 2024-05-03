const addon = require('bindings')('sherpa-onnx-node-addon-api-native');
const streaming_asr = require('./streaming-asr.js');

module.exports = {
  OnlineRecognizer: streaming_asr.OnlineRecognizer,
  readWave: addon.readWave,
}
