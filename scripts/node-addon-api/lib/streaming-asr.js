const addon = require('bindings')('sherpa-onnx-node-addon-api-native');


class OnlineRecognizer {
  constructor(config) {
    this.handle = addon.createOnlineRecognizer(config)
  }
}

module.exports = {OnlineRecognizer}
