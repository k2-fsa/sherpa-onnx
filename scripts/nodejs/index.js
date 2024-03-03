// Copyright (c)  2023-2024  Xiaomi Corporation (authors: Fangjun Kuang)
'use strict'

const wasmModule = require('./sherpa-onnx-wasm-nodejs.js')();
const sherpa_onnx_asr = require('./sherpa-onnx-asr.js');
const sherpa_onnx_tts = require('./sherpa-onnx-tts.js');

function createOnlineRecognizer(config) {
  return sherpa_onnx_asr.createOnlineRecognizer(wasmModule, config);
}

function createOfflineRecognizer(config) {
  return new sherpa_onnx_asr.OfflineRecognizer(config, wasmModule);
}

function createOfflineTts(config) {
  return sherpa_onnx_tts.createOfflineTts(wasmModule, config);
}

// Note: online means streaming and offline means non-streaming here.
// Both of them don't require internet connection.
module.exports = {
  createOnlineRecognizer,
  createOfflineRecognizer,
  createOfflineTts,
};
