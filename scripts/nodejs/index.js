// Copyright (c)  2023  Xiaomi Corporation (authors: Fangjun Kuang)
//
// Please use
//
// npm install ffi-napi ref-struct-napi
//
// before you use this file
//
//
// Please use node 13. node 16, 18, 20, and 21 are known not working.
// See also
// https://github.com/node-ffi-napi/node-ffi-napi/issues/244
// and
// https://github.com/node-ffi-napi/node-ffi-napi/issues/97
'use strict'

const debug = require('debug')('sherpa-onnx');
const os = require('os');
const path = require('path');
const ffi = require('ffi-napi');
const ref = require('ref-napi');
const fs = require('fs');

const StructType = require('ref-struct-napi');
const cstring = ref.types.CString;
const cstringPtr = ref.types(cstring);
const int32_t = ref.types.int32;
const float = ref.types.float;
const floatPtr = ref.refType(float);

const SherpaOnnxOnlineTransducerModelConfig = StructType({
  'encoder': cstring,
  'decoder': cstring,
  'joiner': cstring,
});

const SherpaOnnxOnlineParaformerModelConfig = StructType({
  'encoder': cstring,
  'decoder': cstring,
});

const SherpaOnnxModelConfig = StructType({
  'transducer': SherpaOnnxOnlineTransducerModelConfig,
  'paraformer': SherpaOnnxOnlineParaformerModelConfig,
  'tokens': cstring,
  'numThreads': int32_t,
  'provider': cstring,
  'debug': int32_t,
  'modelType': cstring,
});

const SherpaOnnxFeatureConfig = StructType({
  'sampleRate': int32_t,
  'featureDim': int32_t,
});

const SherpaOnnxOnlineRecognizerConfig = StructType({
  'sampleRate': int32_t,
  'featureDim': int32_t,
  'featConfig': SherpaOnnxFeatureConfig,
  'modelConfig': SherpaOnnxOnlineModelConfig,
  'decodingMethod': cstring,
  'maxActivePaths': int32_t,
  'enableEndpoint': int32_t,
  'rule1MinTrailingSilence': float
  'rule2MinTrailingSilence': float
  'rule3MinUtteranceLength': float,
  'hotwords_file': cstring,
  'hotwordsScore': float,
});

const SherpaOnnxOnlineRecognizerResult = StructType({
  'text': cstring,
  'tokens': cstring,
  'tokensArr': cstringPtr,
  'timestamps': floatPtr,
  'count': int32_t,
  'json': cstring,
});

const SherpaOnnxOnlineRecognizerPtr = ref.refType(ref.types.void);
const SherpaOnnxOnlineStreamPtr = ref.refType(ref.types.void);

const SherpaOnnxOnlineRecognizerConfigPtr =
    ref.refType(SherpaOnnxOnlineRecognizerConfig);

let soname;
if (os.platform() == 'win32') {
  // see https://nodejs.org/api/process.html#processarch
  if (process.arch == 'x64') {
    let currentPath = process.env.Path;
    let dllDirectory = path.resolve(path.join(__dirname, 'lib', 'win-x64'));
    process.env.Path = currentPath + path.delimiter + dllDirectory;

    soname = path.join(__dirname, 'lib', 'win-x64', 'sherpa-onnx-c-api.dll')
  } else if (process.arch == 'ia32') {
    let currentPath = process.env.Path;
    let dllDirectory = path.resolve(path.join(__dirname, 'lib', 'win-x86'));
    process.env.Path = currentPath + path.delimiter + dllDirectory;

    soname = path.join(__dirname, 'lib', 'win-x86', 'sherpa-onnx-c-api.dll')
  } else {
    throw new Error(
        `Support only Windows x86 and x64 for now. Given ${process.arch}`);
  }
} else if (os.platform() == 'darwin') {
  if (process.arch == 'x64') {
    soname =
        path.join(__dirname, 'lib', 'osx-x64', 'libsherpa-onnx-c-api.dylib');
  } else if (process.arch == 'arm64') {
    soname =
        path.join(__dirname, 'lib', 'osx-arm64', 'libsherpa-onnx-c-api.dylib');
  } else {
    throw new Error(
        `Support only macOS x64 and arm64 for now. Given ${process.arch}`);
  }
} else if (os.platform() == 'linux') {
  if (process.arch == 'x64') {
    soname =
        path.join(__dirname, 'lib', 'linux-x64', 'libsherpa-onnx-c-api.so');
  } else if (process.arch == 'ia32') {
    soname =
        path.join(__dirname, 'lib', 'linux-x86', 'libsherpa-onnx-c-api.so');
  } else {
    throw new Error(
        `Support only Linux x86 and x64 for now. Given ${process.arch}`);
  }
} else {
  throw new Error(`Unsupported platform ${os.platform()}`);
}
if (!fs.existsSync(soname)) {
  throw new Error(`Cannot find file ${soname}. Please make sure you have run
      ./build.sh`);
}

debug('soname ', soname)

const onnx = ffi.Library(soname, {
  'CreateOnlineRecognizer':
      [SherpaOnnxOnlineRecognizerPtr, [SherpaOnnxOnlineRecognizerConfigPtr]],

  'DestroyRecognizer': ['void', [RecognizerPtr]],
  'CreateStream': [StreamPtr, [RecognizerPtr]],
  'DestroyStream': ['void', [StreamPtr]],
  'AcceptWaveform': ['void', [StreamPtr, float, floatPtr, int32_t]],
  'IsReady': [int32_t, [RecognizerPtr, StreamPtr]],
  'Decode': ['void', [RecognizerPtr, StreamPtr]],
  'GetResult': [ResultPtr, [RecognizerPtr, StreamPtr]],
  'DestroyResult': ['void', [ResultPtr]],
  'Reset': ['void', [RecognizerPtr, StreamPtr]],
  'InputFinished': ['void', [StreamPtr]],
  'IsEndpoint': [int32_t, [RecognizerPtr, StreamPtr]],
});
