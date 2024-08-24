// Copyright (c)  2023-2024  Xiaomi Corporation (authors: Fangjun Kuang)
'use strict'

const wasmModule = require('./sherpa-onnx-wasm-nodejs.js')();
const sherpa_onnx_asr = require('./sherpa-onnx-asr.js');
const sherpa_onnx_tts = require('./sherpa-onnx-tts.js');
const sherpa_onnx_kws = require('./sherpa-onnx-kws.js');
const sherpa_onnx_wave = require('./sherpa-onnx-wave.js');
const sherpa_onnx_vad = require('./sherpa-onnx-vad.js');

function createOnlineRecognizer(config) {
  return sherpa_onnx_asr.createOnlineRecognizer(wasmModule, config);
}

function createOfflineRecognizer(config) {
  return new sherpa_onnx_asr.OfflineRecognizer(config, wasmModule);
}

function createOfflineTts(config) {
  return sherpa_onnx_tts.createOfflineTts(wasmModule, config);
}

function createKws(config) {
  return sherpa_onnx_kws.createKws(wasmModule, config);
}

function createCircularBuffer(capacity) {
  return new sherpa_onnx_vad.CircularBuffer(capacity, wasmModule);
}

function createVad(config) {
  return sherpa_onnx_vad.createVad(wasmModule, config);
}

function readWave(filename) {
  return sherpa_onnx_wave.readWave(filename, wasmModule);
}

function writeWave(filename, data) {
  sherpa_onnx_wave.writeWave(filename, data, wasmModule);
}

// Note: online means streaming and offline means non-streaming here.
// Both of them don't require internet connection.
module.exports = {
  createOnlineRecognizer,
  createOfflineRecognizer,
  createOfflineTts,
  createKws,
  readWave,
  writeWave,
  createCircularBuffer,
  createVad,
};
