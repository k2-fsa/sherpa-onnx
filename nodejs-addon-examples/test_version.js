// Copyright (c)  2026  Xiaomi Corporation
//
// Print the sherpa-onnx and onnxruntime versions.
//
// Usage:
//   node test_version.js

const sherpa_onnx = require('sherpa-onnx-node');

console.log(`sherpa-onnx version: ${sherpa_onnx.version}`);
console.log(`sherpa-onnx gitSha1: ${sherpa_onnx.gitSha1}`);
console.log(`sherpa-onnx gitDate: ${sherpa_onnx.gitDate}`);
console.log(`onnxruntime version: ${sherpa_onnx.onnxruntimeVersion}`);

if (typeof sherpa_onnx.version !== 'string' || sherpa_onnx.version.length === 0) {
  throw new Error('sherpa_onnx.version is missing or empty');
}
if (typeof sherpa_onnx.onnxruntimeVersion !== 'string' ||
    sherpa_onnx.onnxruntimeVersion.length === 0) {
  throw new Error('sherpa_onnx.onnxruntimeVersion is missing or empty');
}
