const os = require('os');

let addon = null;

const platform = os.platform() === 'win32' ? 'win' : os.platform();
const arch = os.arch();

try {
  if (arch === 'x64') {
    if (platform === 'win') {
      // @ts-expect-error
      addon = require('../sherpa-onnx-win-x64/sherpa-onnx.node')
    } else if (platform === 'darwin') {
      // @ts-expect-error
      addon = require('../sherpa-onnx-darwin-x64/sherpa-onnx.node')
    } else if (platform === 'linux') {
      // @ts-expect-error
      addon = require('../sherpa-onnx-linux-x64/sherpa-onnx.node')
    }
  } else if (arch === 'arm64') {
    if (platform === 'darwin') {
      // @ts-expect-error
      addon = require('../sherpa-onnx-darwin-arm64/sherpa-onnx.node')
    } else if (platform === 'linux') {
      // @ts-expect-error
      addon = require('../sherpa-onnx-linux-arm64/sherpa-onnx.node')
    }
  } else if (arch === 'ia32') {
    if (platform === 'win') {
      // @ts-expect-error
      addon = require('../sherpa-onnx-win-ia32/sherpa-onnx.node')
    }
  }
} catch (error) {
  //
}

if (!addon) {
  try {
    if (arch === 'x64') {
      if (platform === 'win') {
        // @ts-expect-error
        addon = require('./node_modules/sherpa-onnx-win-x64/sherpa-onnx.node')
      } else if (platform === 'darwin') {
        // @ts-expect-error
        addon = require('./node_modules/sherpa-onnx-darwin-x64/sherpa-onnx.node')
      } else if (platform === 'linux') {
        // @ts-expect-error
        addon = require('./node_modules/sherpa-onnx-linux-x64/sherpa-onnx.node')
      }
    } else if (arch === 'arm64') {
      if (platform === 'darwin') {
        // @ts-expect-error
        addon = require('./node_modules/sherpa-onnx-darwin-arm64/sherpa-onnx.node')
      } else if (platform === 'linux') {
        // @ts-expect-error
        addon = require('./node_modules/sherpa-onnx-linux-arm64/sherpa-onnx.node')
      }
    } else if (arch === 'ia32') {
      if (platform === 'win') {
        // @ts-expect-error
        addon = require('./node_modules/sherpa-onnx-win-ia32/sherpa-onnx.node')
      }
    }
  } catch (error) {
    //
  }
}
 
module.exports = addon;