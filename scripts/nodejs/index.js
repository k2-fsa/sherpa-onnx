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
  'hotwordsFile': cstring,
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
const SherpaOnnxOnlineStreamPtrPtr = ref.refType(SherpaOnnxOnlineStreamPtr);
const SherpaOnnxOnlineRecognizerResultPtr = ref.refType(SherpaOnnxOnlineRecognizerResult);

const SherpaOnnxOnlineRecognizerConfigPtr =
    ref.refType(SherpaOnnxOnlineRecognizerConfig);

const SherpaOnnxOfflineTransducerModelConfig = StructType({
  'encoder': cstring,
  'decoder': cstring,
  'joiner': cstring,
});

const SherpaOnnxOfflineParaformerModelConfig = StructType({
  'model': cstring,
});

const SherpaOnnxOfflineNemoEncDecCtcModelConfig = StructType({
  'model': cstring,
});

const SherpaOnnxOfflineWhisperModelConfig = StructType({
  'encoder': cstring,
  'decoder': cstring,
});

const SherpaOnnxOfflineTdnnModelConfig = StructType({
  'model': cstring,
});

const SherpaOnnxOfflineLMConfig = StructType({
  'model': cstring,
  'scale': float,
});

const SherpaOnnxOfflineModelConfig = StructType({
  'transducer': SherpaOnnxOfflineTransducerModelConfig,
  'paraformer': SherpaOnnxOfflineParaformerModelConfig,
  'nemoCtc': SherpaOnnxOfflineNemoEncDecCtcModelConfig,
  'whisper': SherpaOnnxOfflineWhisperModelConfig,
  'tdnn': SherpaOnnxOfflineTdnnModelConfig,
  'tokens': cstring,
  'numThreads': int32_t,
  'debug': int32_t,
  'provider': cstring,
  'modelType': cstring,
});

const SherpaOnnxOfflineModelConfig = StructType({
  'featConfig': SherpaOnnxFeatureConfig,
  'modelConfig': SherpaOnnxOfflineModelConfig,
  'lmConfig': SherpaOnnxOfflineLMConfig,
  'decodingMethod': cstring,
  'maxActivePaths': int32_t,
  'hotwordsFile': cstring,
  'hotwordsScore': float,
});

const SherpaOnnxOfflineModelConfig = StructType({
  'text': cstring,
  'timestamps': floatPtr,,
  'count': int32_t,,

});

const SherpaOnnxOfflineRecognizerPtr = ref.refType(ref.types.void);
const SherpaOnnxOffineStreamPtr = ref.refType(ref.types.void);
const SherpaOnnxOffineStreamPtrPtr = ref.refType(SherpaOnnxOfflineStreamPtr);
const SherpaOnnxOffineRecognizerResultPtr = ref.refType(SherpaOnnxOfflineRecognizerResult);

const SherpaOnnxOfflineRecognizerConfigPtr =
    ref.refType(SherpaOnnxOfflineRecognizerConfig);

// vad
const SherpaOnnxSileroVadModelConfig = StructType({
  'model': cstring,
  'threshold': float,
  'minSilenceDuration': float,
  'minSpeechDuration': float,
  'windowSize': int32_t,
});

const SherpaOnnxVadModelConfig = StructType({
  'sileroVad': SherpaOnnxSileroVadModelConfig,
  'sampleRate': int32_t,
  'numThreads': int32_t,
  'provider': cstring,
  'debug': int32_t,
});

const SherpaOnnxSpeechSegment = StructType({
  'start': int32_t,
  'samples': floatPtr,
  'n': int32_t,
});

const SherpaOnnxSileroVadModelConfigPtr= ref.refType(SherpaOnnxSileroVadModelConfig);
const SherpaOnnxSpeechSegmentPtr= ref.refType(SherpaOnnxSpeechSegment);
const SherpaOnnxCircularBufferPtr = ref.refType(ref.types.void);
const SherpaOnnxVoiceActivityDetectorPtr = ref.refType(ref.types.void);

// tts
const SherpaOnnxOfflineTtsVitsModelConfig = StructType({
  'model': cstring,
  'lexicon': cstring,
  'tokens': cstring,
  'noiseScale': float,
  'noiseScaleW': float,
  'lengthScale': float,
});

const SherpaOnnxOfflineTtsConfig = StructType({
  'model': SherpaOnnxOfflineTtsModelConfig,
});

const SherpaOnnxGeneratedAudio = StructType({
  'samples': floatPtr,
  'n': int32_t,
  'sampleRate': int32_t,
});

const SherpaOnnxOfflineTtsVitsModelConfigPtr = ref.refType(SherpaOnnxOfflineTtsVitsModelConfig);
const SherpaOnnxOfflineTtsConfigPtr = ref.refType(SherpaOnnxOfflineTtsConfig);
const SherpaOnnxGeneratedAudioPtr = ref.refType(SherpaOnnxGeneratedAudio);
const SherpaOnnxOfflineTtsPtr = ref.refType(SherpaOnnxOfflineTts);

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
  // online asr
  'CreateOnlineRecognizer':
      [SherpaOnnxOnlineRecognizerPtr, [SherpaOnnxOnlineRecognizerConfigPtr]],
  'DestroyOnlineRecognizer': ['void', [SherpaOnnxOnlineRecognizerPtr]],
  'CreateOnlineStream': [SherpaOnnxOnlineStreamPtr, [SherpaOnnxOnlineRecognizerPtr]],
  'CreateOnlineStreamWithHotwords': [SherpaOnnxOnlineStreamPtr, [SherpaOnnxOnlineRecognizerPtr, cstring]],
  'DestroyOnlineStream': ['void', [SherpaOnnxOnlineStreamPtr]],
  'AcceptWaveform': ['void', [SherpaOnnxOnlineStreamPtr, int32_t, floatPtr, int32_t]],
  'IsOnlineStreamReady': [int32_t, [SherpaOnnxOnlineRecognizerPtr, SherpaOnnxOnlineStreamPtr]],
  'DecodeOnlineStream': ['void', [SherpaOnnxOnlineRecognizerPtr, SherpaOnnxOnlineStreamPtr]],
  'DecodeMultipleOnlineStreams': ['void', [SherpaOnnxOnlineRecognizerPtr, SherpaOnnxOnlineStreamPtrPtr, int32_t]],
  'GetOnlineStreamResult': ['SherpaOnnxOnlineRecognizerResultPtr', [SherpaOnnxOnlineRecognizerPtr,SherpaOnnxOnlineStreamPtr]],
  'DestroyOnlineRecognizerResult': ['void', [SherpaOnnxOnlineRecognizerResultPtr]],
  'Reset': ['void', [SherpaOnnxOnlineRecognizerPtr, SherpaOnnxOnlineStreamPtr]],
  'InputFinished': ['void', [SherpaOnnxOnlineStreamPtr]],
  'IsEndpoint': [int32_t, [SherpaOnnxOnlineRecognizerPtr, SherpaOnnxOnlineStreamPtr]],

  // offline asr
  'CreateOfflineRecognizer':
      [SherpaOnnxOfflineRecognizerPtr, [SherpaOnnxOfflineRecognizerConfigPtr]],
  'DestroyOfflineRecognizer': ['void', [SherpaOnnxOfflineRecognizerPtr]],
  'CreateOfflineStream': [SherpaOnnxOfflineStreamPtr, [SherpaOnnxOfflineRecognizerPtr]],
  'DestroyOfflineStream': ['void', [SherpaOnnxOfflineStreamPtr]],
  'AcceptWaveformOffline': ['void', [SherpaOnnxOfflineStreamPtr, int32_t, floatPtr, int32_t]],
  'DecodeOfflineStream': ['void', [SherpaOnnxOfflineRecognizerPtr, SherpaOnnxOfflineStreamPtr]],
  'DecodeMultipleOfflineStreams': ['void', [SherpaOnnxOfflineRecognizerPtr, SherpaOnnxOfflineStreamPtrPtr, int32_t]],
  'GetOfflineStreamResult': ['SherpaOnnxOfflineRecognizerResultPtr', [SherpaOnnxOfflineStreamPtr]],
  'DestroyOfflineRecognizerResult': ['void', [SherpaOnnxOfflineRecognizerResultPtr]],

  // vad
  'SherpaOnnxCreateCircularBuffer': [SherpaOnnxCircularBufferPtr, [int32_t]],
  'SherpaOnnxDestroyCircularBuffer': ['void', [SherpaOnnxCircularBufferPtr]],
  'SherpaOnnxCircularBufferPush': ['void', [SherpaOnnxCircularBufferPtr, floatPtr, int32_t]],
  'SherpaOnnxCircularBufferGet': [floatPtr, [SherpaOnnxCircularBufferPtr, int32_t, int32_t]],
  'SherpaOnnxCircularBufferFree': ['void', [floatPtr]],
  'SherpaOnnxCircularBufferPop': ['void', [SherpaOnnxCircularBufferPtr, int32_t]],
  'SherpaOnnxCircularBufferSize': [int32_t, [SherpaOnnxCircularBufferPtr]],
  'SherpaOnnxCircularBufferReset': ['void', [SherpaOnnxCircularBufferPtr]],
  'SherpaOnnxCreateVoiceActivityDetector': [SherpaOnnxVoiceActivityDetectorPtr, [SherpaOnnxVadModelConfigPtr, float]],
  'SherpaOnnxDestroyVoiceActivityDetector': ['void', [SherpaOnnxVoiceActivityDetectorPtr]],
  'SherpaOnnxVoiceActivityDetectorAcceptWaveform': ['void', [SherpaOnnxVoiceActivityDetectorPtr, floatPtr, int32_t]],
  'SherpaOnnxVoiceActivityDetectorEmpty': [int32_t, [SherpaOnnxVoiceActivityDetectorPtr]],
  'SherpaOnnxVoiceActivityDetectorDetected': [int32_t, [SherpaOnnxVoiceActivityDetectorPtr]],
  'SherpaOnnxVoiceActivityDetectorPop': ['void', [SherpaOnnxVoiceActivityDetectorPtr]],
  'SherpaOnnxVoiceActivityDetectorClear': ['void', [SherpaOnnxVoiceActivityDetectorPtr]],
  'SherpaOnnxVoiceActivityDetectorFront': [SherpaOnnxSpeechSegmentPtr, [SherpaOnnxVoiceActivityDetectorPtr]],
  'SherpaOnnxDestroySpeechSegment': ['void', [SherpaOnnxSpeechSegmentPtr]],
  'SherpaOnnxVoiceActivityDetectorReset': ['void', [SherpaOnnxVoiceActivityDetectorPtr]],
  // tts
  'SherpaOnnxCreateOfflineTts': [SherpaOnnxOfflineTtsPtr, [SherpaOnnxOfflineTtsConfigPtr]],
  'SherpaOnnxDestroyOfflineTts': ['void', [SherpaOnnxOfflineTtsPtr]],
  'SherpaOnnxOfflineTtsGenerate': [SherpaOnnxGeneratedAudioPtr, [SherpaOnnxOfflineTtsPtr, cstr, int32_t, float]],
  'SherpaOnnxDestroyOfflineTtsGeneratedAudio': ['void', [SherpaOnnxGeneratedAudioPtr]],
  'SherpaOnnxWriteWave': ['void', [floatPtr, int32_t, int32_t, cstring]],
});
