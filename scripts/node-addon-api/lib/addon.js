/** @typedef {import('./types').WaveObject} WaveObject */

const os = require('os');
const path = require('path');
const addonStaticImport = require('./addon-static-import');

// Package name triggered spam for sherpa-onnx-win32-x64
// so we have renamed it to sherpa-onnx-win-x64
const platform = os.platform() === 'win32' ? 'win' : os.platform();
const arch = os.arch();
const platform_arch = `${platform}-${arch}`;
const possible_paths = [
  '../build/Release/sherpa-onnx.node',
  '../build/Debug/sherpa-onnx.node',
  `./node_modules/sherpa-onnx-${platform_arch}/sherpa-onnx.node`,
  `../sherpa-onnx-${platform_arch}/sherpa-onnx.node`,
  './sherpa-onnx.node',
];

let addon = addonStaticImport;

if (!addon) {
  for (const p of possible_paths) {
    try {
      addon = require(p);
      break;
    } catch (error) {
      // do nothing; try the next option
      ;
    }
  }
}

module.exports = addon;

if (!addon) {
  let addon_path =
      `${process.env.PWD}/node_modules/sherpa-onnx-${platform_arch}`;
  const pnpmIndex = __dirname.indexOf(`node_modules${path.sep}.pnpm`);
  if (pnpmIndex !== -1) {
    const parts = __dirname.slice(pnpmIndex).split(path.sep);
    parts.pop();
    addon_path =
        `${process.env.PWD}/${parts.join('/')}/sherpa-onnx-${platform_arch}`;
  }

  let msg = `Could not find sherpa-onnx-node. Tried\n\n  ${
      possible_paths.join('\n  ')}\n`
  if (os.platform() == 'darwin' &&
      (!process.env.DYLD_LIBRARY_PATH ||
       !process.env.DYLD_LIBRARY_PATH.includes(
           `node_modules/sherpa-onnx-${platform_arch}`))) {
    msg +=
        'Please remeber to set the following environment variable and try again:\n';

    msg += `export DYLD_LIBRARY_PATH=${addon_path}`;

    msg += ':$DYLD_LIBRARY_PATH\n';
  }

  if (os.platform() == 'linux' &&
      (!process.env.LD_LIBRARY_PATH ||
       !process.env.LD_LIBRARY_PATH.includes(
           `node_modules/sherpa-onnx-${platform_arch}`))) {
    msg +=
        'Please remeber to set the following environment variable and try again:\n';

    msg += `export LD_LIBRARY_PATH=${addon_path}`;

    msg += ':$LD_LIBRARY_PATH\n';
  }

  throw new Error(msg)
}

/**
 * Read a wave file from disk.
 * @function module.exports.readWave
 * @param {string} filename
 * @param {boolean} [enableExternalBuffer=true]
 * @returns {WaveObject}
 */

/**
 * Read a wave from binary buffer.
 * @function module.exports.readWaveFromBinary
 * @param {Uint8Array} data - Binary contents of a wave file.
 * @param {boolean} [enableExternalBuffer=true]
 * @returns {WaveObject}
 */

/**
 * Write a wave file to disk.
 * @function module.exports.writeWave
 * @param {string} filename
 * @param {WaveObject} obj - { samples: Float32Array, sampleRate: number }
 * @returns {boolean}
 */
