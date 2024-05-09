const addon = require('./addon.js')
const streaming_asr = require('./streaming-asr.js');

module.exports = {
  OnlineRecognizer: streaming_asr.OnlineRecognizer,
  readWave: addon.readWave,
}
