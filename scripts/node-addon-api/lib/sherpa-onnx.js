const addon = require('./addon.js')
const streaming_asr = require('./streaming-asr.js');
const vad = require('./vad.js');

module.exports = {
  OnlineRecognizer: streaming_asr.OnlineRecognizer,
  readWave: addon.readWave,
  writeWave: addon.writeWave,
  Display: streaming_asr.Display,
  Vad: vad.Vad,
  CircularBuffer: vad.CircularBuffer,
}
