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
"use strict"

const debug = require("debug")("sherpa-onnx");
const os = require("os");
const path = require("path");
const ffi = require("ffi-napi");
const ref = require("ref-napi");
const fs = require("fs");
var ArrayType = require("ref-array-napi");

const FloatArray = ArrayType(ref.types.float);
const StructType = require("ref-struct-napi");
const cstring = ref.types.CString;
const cstringPtr = ref.refType(cstring);
const int32_t = ref.types.int32;
const float = ref.types.float;
const floatPtr = ref.refType(float);

const SherpaOnnxOnlineTransducerModelConfig = StructType({
  "encoder" : cstring,
  "decoder" : cstring,
  "joiner" : cstring,
});

const SherpaOnnxOnlineParaformerModelConfig = StructType({
  "encoder" : cstring,
  "decoder" : cstring,
});

const SherpaOnnxOnlineZipformer2CtcModelConfig = StructType({
  "model" : cstring,
});

const SherpaOnnxOnlineModelConfig = StructType({
  "transducer" : SherpaOnnxOnlineTransducerModelConfig,
  "paraformer" : SherpaOnnxOnlineParaformerModelConfig,
  "zipformer2Ctc" : SherpaOnnxOnlineZipformer2CtcModelConfig,
  "tokens" : cstring,
  "numThreads" : int32_t,
  "provider" : cstring,
  "debug" : int32_t,
  "modelType" : cstring,
});

const SherpaOnnxFeatureConfig = StructType({
  "sampleRate" : int32_t,
  "featureDim" : int32_t,
});

const SherpaOnnxOnlineRecognizerConfig = StructType({
  "featConfig" : SherpaOnnxFeatureConfig,
  "modelConfig" : SherpaOnnxOnlineModelConfig,
  "decodingMethod" : cstring,
  "maxActivePaths" : int32_t,
  "enableEndpoint" : int32_t,
  "rule1MinTrailingSilence" : float,
  "rule2MinTrailingSilence" : float,
  "rule3MinUtteranceLength" : float,
  "hotwordsFile" : cstring,
  "hotwordsScore" : float,
});

const SherpaOnnxOnlineRecognizerResult = StructType({
  "text" : cstring,
  "tokens" : cstring,
  "tokensArr" : cstringPtr,
  "timestamps" : floatPtr,
  "count" : int32_t,
  "json" : cstring,
});

const SherpaOnnxOnlineRecognizerPtr = ref.refType(ref.types.void);
const SherpaOnnxOnlineStreamPtr = ref.refType(ref.types.void);
const SherpaOnnxOnlineStreamPtrPtr = ref.refType(SherpaOnnxOnlineStreamPtr);
const SherpaOnnxOnlineRecognizerResultPtr =
    ref.refType(SherpaOnnxOnlineRecognizerResult);

const SherpaOnnxOnlineRecognizerConfigPtr =
    ref.refType(SherpaOnnxOnlineRecognizerConfig);

const SherpaOnnxOfflineTransducerModelConfig = StructType({
  "encoder" : cstring,
  "decoder" : cstring,
  "joiner" : cstring,
});

const SherpaOnnxOfflineParaformerModelConfig = StructType({
  "model" : cstring,
});

const SherpaOnnxOfflineNemoEncDecCtcModelConfig = StructType({
  "model" : cstring,
});

const SherpaOnnxOfflineWhisperModelConfig = StructType({
  "encoder" : cstring,
  "decoder" : cstring,
});

const SherpaOnnxOfflineTdnnModelConfig = StructType({
  "model" : cstring,
});

const SherpaOnnxOfflineLMConfig = StructType({
  "model" : cstring,
  "scale" : float,
});

const SherpaOnnxOfflineModelConfig = StructType({
  "transducer" : SherpaOnnxOfflineTransducerModelConfig,
  "paraformer" : SherpaOnnxOfflineParaformerModelConfig,
  "nemoCtc" : SherpaOnnxOfflineNemoEncDecCtcModelConfig,
  "whisper" : SherpaOnnxOfflineWhisperModelConfig,
  "tdnn" : SherpaOnnxOfflineTdnnModelConfig,
  "tokens" : cstring,
  "numThreads" : int32_t,
  "debug" : int32_t,
  "provider" : cstring,
  "modelType" : cstring,
});

const SherpaOnnxOfflineRecognizerConfig = StructType({
  "featConfig" : SherpaOnnxFeatureConfig,
  "modelConfig" : SherpaOnnxOfflineModelConfig,
  "lmConfig" : SherpaOnnxOfflineLMConfig,
  "decodingMethod" : cstring,
  "maxActivePaths" : int32_t,
  "hotwordsFile" : cstring,
  "hotwordsScore" : float,
});

const SherpaOnnxOfflineRecognizerResult = StructType({
  "text" : cstring,
  "timestamps" : floatPtr,
  "count" : int32_t,
});

const SherpaOnnxOfflineRecognizerPtr = ref.refType(ref.types.void);
const SherpaOnnxOfflineStreamPtr = ref.refType(ref.types.void);
const SherpaOnnxOfflineStreamPtrPtr = ref.refType(SherpaOnnxOfflineStreamPtr);
const SherpaOnnxOfflineRecognizerResultPtr =
    ref.refType(SherpaOnnxOfflineRecognizerResult);

const SherpaOnnxOfflineRecognizerConfigPtr =
    ref.refType(SherpaOnnxOfflineRecognizerConfig);

// vad
const SherpaOnnxSileroVadModelConfig = StructType({
  "model" : cstring,
  "threshold" : float,
  "minSilenceDuration" : float,
  "minSpeechDuration" : float,
  "windowSize" : int32_t,
});

const SherpaOnnxVadModelConfig = StructType({
  "sileroVad" : SherpaOnnxSileroVadModelConfig,
  "sampleRate" : int32_t,
  "numThreads" : int32_t,
  "provider" : cstring,
  "debug" : int32_t,
});

const SherpaOnnxSpeechSegment = StructType({
  "start" : int32_t,
  "samples" : FloatArray,
  "n" : int32_t,
});

const SherpaOnnxVadModelConfigPtr = ref.refType(SherpaOnnxVadModelConfig);
const SherpaOnnxSpeechSegmentPtr = ref.refType(SherpaOnnxSpeechSegment);
const SherpaOnnxCircularBufferPtr = ref.refType(ref.types.void);
const SherpaOnnxVoiceActivityDetectorPtr = ref.refType(ref.types.void);

// tts
const SherpaOnnxOfflineTtsVitsModelConfig = StructType({
  "model" : cstring,
  "lexicon" : cstring,
  "tokens" : cstring,
  "dataDir" : cstring,
  "noiseScale" : float,
  "noiseScaleW" : float,
  "lengthScale" : float,
});

const SherpaOnnxOfflineTtsModelConfig = StructType({
  "vits" : SherpaOnnxOfflineTtsVitsModelConfig,
  "numThreads" : int32_t,
  "debug" : int32_t,
  "provider" : cstring,
});

const SherpaOnnxOfflineTtsConfig = StructType({
  "model" : SherpaOnnxOfflineTtsModelConfig,
  "ruleFsts" : cstring,
  "maxNumSentences" : int32_t,
});

const SherpaOnnxGeneratedAudio = StructType({
  "samples" : FloatArray,
  "n" : int32_t,
  "sampleRate" : int32_t,
});

const SherpaOnnxOfflineTtsVitsModelConfigPtr =
    ref.refType(SherpaOnnxOfflineTtsVitsModelConfig);
const SherpaOnnxOfflineTtsConfigPtr = ref.refType(SherpaOnnxOfflineTtsConfig);
const SherpaOnnxGeneratedAudioPtr = ref.refType(SherpaOnnxGeneratedAudio);
const SherpaOnnxOfflineTtsPtr = ref.refType(ref.types.void);

const SherpaOnnxDisplayPtr = ref.refType(ref.types.void);

let soname;
if (os.platform() == "win32") {
  // see https://nodejs.org/api/process.html#processarch
  if (process.arch == "x64") {
    let currentPath = process.env.Path;
    let dllDirectory = path.resolve(path.join(__dirname, "lib", "win-x64"));
    process.env.Path = currentPath + path.delimiter + dllDirectory;

    soname = path.join(__dirname, "lib", "win-x64", "sherpa-onnx-c-api.dll")
  } else if (process.arch == "ia32") {
    let currentPath = process.env.Path;
    let dllDirectory = path.resolve(path.join(__dirname, "lib", "win-x86"));
    process.env.Path = currentPath + path.delimiter + dllDirectory;

    soname = path.join(__dirname, "lib", "win-x86", "sherpa-onnx-c-api.dll")
  } else {
    throw new Error(
        `Support only Windows x86 and x64 for now. Given ${process.arch}`);
  }
} else if (os.platform() == "darwin") {
  if (process.arch == "x64") {
    soname =
        path.join(__dirname, "lib", "osx-x64", "libsherpa-onnx-c-api.dylib");
  } else if (process.arch == "arm64") {
    soname =
        path.join(__dirname, "lib", "osx-arm64", "libsherpa-onnx-c-api.dylib");
  } else {
    throw new Error(
        `Support only macOS x64 and arm64 for now. Given ${process.arch}`);
  }
} else if (os.platform() == "linux") {
  if (process.arch == "x64") {
    soname =
        path.join(__dirname, "lib", "linux-x64", "libsherpa-onnx-c-api.so");
  } else {
    throw new Error(`Support only Linux x64 for now. Given ${process.arch}`);
  }
} else {
  throw new Error(`Unsupported platform ${os.platform()}`);
}

if (!fs.existsSync(soname)) {
  throw new Error(`Cannot find file ${soname}. Please make sure you have run
      ./build.sh`);
}

debug("soname ", soname)

const libsherpa_onnx = ffi.Library(soname, {
  // online asr
  "CreateOnlineRecognizer" : [
    SherpaOnnxOnlineRecognizerPtr, [ SherpaOnnxOnlineRecognizerConfigPtr ]
  ],
  "DestroyOnlineRecognizer" : [ "void", [ SherpaOnnxOnlineRecognizerPtr ] ],
  "CreateOnlineStream" :
      [ SherpaOnnxOnlineStreamPtr, [ SherpaOnnxOnlineRecognizerPtr ] ],
  "CreateOnlineStreamWithHotwords" :
      [ SherpaOnnxOnlineStreamPtr, [ SherpaOnnxOnlineRecognizerPtr, cstring ] ],
  "DestroyOnlineStream" : [ "void", [ SherpaOnnxOnlineStreamPtr ] ],
  "AcceptWaveform" :
      [ "void", [ SherpaOnnxOnlineStreamPtr, int32_t, floatPtr, int32_t ] ],
  "IsOnlineStreamReady" :
      [ int32_t, [ SherpaOnnxOnlineRecognizerPtr, SherpaOnnxOnlineStreamPtr ] ],
  "DecodeOnlineStream" :
      [ "void", [ SherpaOnnxOnlineRecognizerPtr, SherpaOnnxOnlineStreamPtr ] ],
  "DecodeMultipleOnlineStreams" : [
    "void",
    [ SherpaOnnxOnlineRecognizerPtr, SherpaOnnxOnlineStreamPtrPtr, int32_t ]
  ],
  "GetOnlineStreamResult" : [
    SherpaOnnxOnlineRecognizerResultPtr,
    [ SherpaOnnxOnlineRecognizerPtr, SherpaOnnxOnlineStreamPtr ]
  ],
  "DestroyOnlineRecognizerResult" :
      [ "void", [ SherpaOnnxOnlineRecognizerResultPtr ] ],
  "Reset" :
      [ "void", [ SherpaOnnxOnlineRecognizerPtr, SherpaOnnxOnlineStreamPtr ] ],
  "InputFinished" : [ "void", [ SherpaOnnxOnlineStreamPtr ] ],
  "IsEndpoint" :
      [ int32_t, [ SherpaOnnxOnlineRecognizerPtr, SherpaOnnxOnlineStreamPtr ] ],

  // offline asr
  "CreateOfflineRecognizer" : [
    SherpaOnnxOfflineRecognizerPtr, [ SherpaOnnxOfflineRecognizerConfigPtr ]
  ],
  "DestroyOfflineRecognizer" : [ "void", [ SherpaOnnxOfflineRecognizerPtr ] ],
  "CreateOfflineStream" :
      [ SherpaOnnxOfflineStreamPtr, [ SherpaOnnxOfflineRecognizerPtr ] ],
  "DestroyOfflineStream" : [ "void", [ SherpaOnnxOfflineStreamPtr ] ],
  "AcceptWaveformOffline" :
      [ "void", [ SherpaOnnxOfflineStreamPtr, int32_t, floatPtr, int32_t ] ],
  "DecodeOfflineStream" : [
    "void", [ SherpaOnnxOfflineRecognizerPtr, SherpaOnnxOfflineStreamPtr ]
  ],
  "DecodeMultipleOfflineStreams" : [
    "void",
    [ SherpaOnnxOfflineRecognizerPtr, SherpaOnnxOfflineStreamPtrPtr, int32_t ]
  ],
  "GetOfflineStreamResult" :
      [ SherpaOnnxOfflineRecognizerResultPtr, [ SherpaOnnxOfflineStreamPtr ] ],
  "DestroyOfflineRecognizerResult" :
      [ "void", [ SherpaOnnxOfflineRecognizerResultPtr ] ],

  // vad
  "SherpaOnnxCreateCircularBuffer" :
      [ SherpaOnnxCircularBufferPtr, [ int32_t ] ],
  "SherpaOnnxDestroyCircularBuffer" :
      [ "void", [ SherpaOnnxCircularBufferPtr ] ],
  "SherpaOnnxCircularBufferPush" :
      [ "void", [ SherpaOnnxCircularBufferPtr, floatPtr, int32_t ] ],
  "SherpaOnnxCircularBufferGet" :
      [ FloatArray, [ SherpaOnnxCircularBufferPtr, int32_t, int32_t ] ],
  "SherpaOnnxCircularBufferFree" : [ "void", [ FloatArray ] ],
  "SherpaOnnxCircularBufferPop" :
      [ "void", [ SherpaOnnxCircularBufferPtr, int32_t ] ],
  "SherpaOnnxCircularBufferSize" : [ int32_t, [ SherpaOnnxCircularBufferPtr ] ],
  "SherpaOnnxCircularBufferHead" : [ int32_t, [ SherpaOnnxCircularBufferPtr ] ],
  "SherpaOnnxCircularBufferReset" : [ "void", [ SherpaOnnxCircularBufferPtr ] ],
  "SherpaOnnxCreateVoiceActivityDetector" : [
    SherpaOnnxVoiceActivityDetectorPtr, [ SherpaOnnxVadModelConfigPtr, float ]
  ],
  "SherpaOnnxDestroyVoiceActivityDetector" :
      [ "void", [ SherpaOnnxVoiceActivityDetectorPtr ] ],
  "SherpaOnnxVoiceActivityDetectorAcceptWaveform" :
      [ "void", [ SherpaOnnxVoiceActivityDetectorPtr, floatPtr, int32_t ] ],
  "SherpaOnnxVoiceActivityDetectorEmpty" :
      [ int32_t, [ SherpaOnnxVoiceActivityDetectorPtr ] ],
  "SherpaOnnxVoiceActivityDetectorDetected" :
      [ int32_t, [ SherpaOnnxVoiceActivityDetectorPtr ] ],
  "SherpaOnnxVoiceActivityDetectorPop" :
      [ "void", [ SherpaOnnxVoiceActivityDetectorPtr ] ],
  "SherpaOnnxVoiceActivityDetectorClear" :
      [ "void", [ SherpaOnnxVoiceActivityDetectorPtr ] ],
  "SherpaOnnxVoiceActivityDetectorFront" :
      [ SherpaOnnxSpeechSegmentPtr, [ SherpaOnnxVoiceActivityDetectorPtr ] ],
  "SherpaOnnxDestroySpeechSegment" : [ "void", [ SherpaOnnxSpeechSegmentPtr ] ],
  "SherpaOnnxVoiceActivityDetectorReset" :
      [ "void", [ SherpaOnnxVoiceActivityDetectorPtr ] ],
  // tts
  "SherpaOnnxCreateOfflineTts" :
      [ SherpaOnnxOfflineTtsPtr, [ SherpaOnnxOfflineTtsConfigPtr ] ],
  "SherpaOnnxDestroyOfflineTts" : [ "void", [ SherpaOnnxOfflineTtsPtr ] ],
  "SherpaOnnxOfflineTtsGenerate" : [
    SherpaOnnxGeneratedAudioPtr,
    [ SherpaOnnxOfflineTtsPtr, cstring, int32_t, float ]
  ],
  "SherpaOnnxDestroyOfflineTtsGeneratedAudio" :
      [ "void", [ SherpaOnnxGeneratedAudioPtr ] ],
  "SherpaOnnxWriteWave" : [ "void", [ floatPtr, int32_t, int32_t, cstring ] ],

  // display
  "CreateDisplay" : [ SherpaOnnxDisplayPtr, [ int32_t ] ],
  "DestroyDisplay" : [ "void", [ SherpaOnnxDisplayPtr ] ],
  "SherpaOnnxPrint" : [ "void", [ SherpaOnnxDisplayPtr, int32_t, cstring ] ],
});

class Display {
  constructor(maxWordPerLine) {
    this.handle = libsherpa_onnx.CreateDisplay(maxWordPerLine);
  }
  free() {
    if (this.handle) {
      libsherpa_onnx.DestroyDisplay(this.handle);
      this.handle = null;
    }
  }

  print(idx, s) { libsherpa_onnx.SherpaOnnxPrint(this.handle, idx, s); }
};

class OnlineResult {
  constructor(text) { this.text = Buffer.from(text, "utf-8").toString(); }
};

class OnlineStream {
  constructor(handle) { this.handle = handle }

  free() {
    if (this.handle) {
      libsherpa_onnx.DestroyOnlineStream(this.handle);
      this.handle = null;
    }
  }

  /**
   * @param sampleRate {Number}
   * @param samples {Float32Array} Containing samples in the range [-1, 1]
   */
  acceptWaveform(sampleRate, samples) {
    libsherpa_onnx.AcceptWaveform(this.handle, sampleRate, samples,
                                  samples.length);
  }
};

class OnlineRecognizer {
  constructor(config) {
    this.config = config;
    this.recognizer_handle =
        libsherpa_onnx.CreateOnlineRecognizer(config.ref());
  }

  free() {
    if (this.recognizer_handle) {
      libsherpa_onnx.DestroyOnlineRecognizer(this.recognizer_handle);
      this.recognizer_handle = null;
    }
  }

  createStream() {
    let handle = libsherpa_onnx.CreateOnlineStream(this.recognizer_handle);
    return new OnlineStream(handle);
  }

  isReady(stream) {
    return libsherpa_onnx.IsOnlineStreamReady(this.recognizer_handle,
                                              stream.handle)
  }

  isEndpoint(stream) {
    return libsherpa_onnx.IsEndpoint(this.recognizer_handle, stream.handle);
  }

  reset(stream) { libsherpa_onnx.Reset(this.recognizer_handle, stream.handle); }

  decode(stream) {
    libsherpa_onnx.DecodeOnlineStream(this.recognizer_handle, stream.handle)
  }

  getResult(stream) {
    let handle = libsherpa_onnx.GetOnlineStreamResult(this.recognizer_handle,
                                                      stream.handle);
    let r = handle.deref();
    let ans = new OnlineResult(r.text);
    libsherpa_onnx.DestroyOnlineRecognizerResult(handle);

    return ans
  }
};

class OfflineResult {
  constructor(text) { this.text = Buffer.from(text, "utf-8").toString(); }
};

class OfflineStream {
  constructor(handle) { this.handle = handle }

  free() {
    if (this.handle) {
      libsherpa_onnx.DestroyOfflineStream(this.handle);
      this.handle = null;
    }
  }

  /**
   * @param sampleRate {Number}
   * @param samples {Float32Array} Containing samples in the range [-1, 1]
   */
  acceptWaveform(sampleRate, samples) {
    libsherpa_onnx.AcceptWaveformOffline(this.handle, sampleRate, samples,
                                         samples.length);
  }
};

class OfflineRecognizer {
  constructor(config) {
    this.config = config;
    this.recognizer_handle =
        libsherpa_onnx.CreateOfflineRecognizer(config.ref());
  }

  free() {
    if (this.recognizer_handle) {
      libsherpa_onnx.DestroyOfflineRecognizer(this.recognizer_handle);
      this.recognizer_handle = null;
    }
  }

  createStream() {
    let handle = libsherpa_onnx.CreateOfflineStream(this.recognizer_handle);
    return new OfflineStream(handle);
  }

  decode(stream) {
    libsherpa_onnx.DecodeOfflineStream(this.recognizer_handle, stream.handle)
  }

  getResult(stream) {
    let handle = libsherpa_onnx.GetOfflineStreamResult(stream.handle);
    let r = handle.deref();
    let ans = new OfflineResult(r.text);
    libsherpa_onnx.DestroyOfflineRecognizerResult(handle);

    return ans
  }
};

class SpeechSegment {
  constructor(start, samples) {
    this.start = start;
    this.samples = samples;
  }
};

// this buffer holds only float entries.
class CircularBuffer {
  /**
   * @param capacity {int} The capacity of the circular buffer.
   */
  constructor(capacity) {
    this.handle = libsherpa_onnx.SherpaOnnxCreateCircularBuffer(capacity);
  }

  free() {
    if (this.handle) {
      libsherpa_onnx.SherpaOnnxDestroyCircularBuffer(this.handle);
      this.handle = null;
    }
  }

  /**
   * @param samples {Float32Array}
   */
  push(samples) {
    libsherpa_onnx.SherpaOnnxCircularBufferPush(this.handle, samples,
                                                samples.length);
  }

  get(startIndex, n) {
    let data =
        libsherpa_onnx.SherpaOnnxCircularBufferGet(this.handle, startIndex, n);

    // https://tootallnate.github.io/ref/#exports-reinterpret
    const buffer = data.buffer.reinterpret(n * ref.sizeof.float).buffer;

    // create a copy since we are going to free the buffer at the end
    let s = new Float32Array(buffer).slice(0);
    libsherpa_onnx.SherpaOnnxCircularBufferFree(data);
    return s;
  }

  pop(n) { libsherpa_onnx.SherpaOnnxCircularBufferPop(this.handle, n); }

  size() { return libsherpa_onnx.SherpaOnnxCircularBufferSize(this.handle); }

  head() { return libsherpa_onnx.SherpaOnnxCircularBufferHead(this.handle); }

  reset() { libsherpa_onnx.SherpaOnnxCircularBufferReset(this.handle); }
};

class VoiceActivityDetector {
  constructor(config, bufferSizeInSeconds) {
    this.config = config;
    this.handle = libsherpa_onnx.SherpaOnnxCreateVoiceActivityDetector(
        config.ref(), bufferSizeInSeconds);
  }

  free() {
    if (this.handle) {
      libsherpa_onnx.SherpaOnnxDestroyVoiceActivityDetector(this.handle);
    }
  }

  acceptWaveform(samples) {
    libsherpa_onnx.SherpaOnnxVoiceActivityDetectorAcceptWaveform(
        this.handle, samples, samples.length);
  }

  isEmpty() {
    return libsherpa_onnx.SherpaOnnxVoiceActivityDetectorEmpty(this.handle);
  }

  isDetected() {
    return libsherpa_onnx.SherpaOnnxVoiceActivityDetectorDetected(this.handle);
  }
  pop() { libsherpa_onnx.SherpaOnnxVoiceActivityDetectorPop(this.handle); }

  clear() { libsherpa_onnx.SherpaOnnxVoiceActivityDetectorClear(this.handle); }

  reset() { libsherpa_onnx.SherpaOnnxVoiceActivityDetectorReset(this.handle); }

  front() {
    let segment =
        libsherpa_onnx.SherpaOnnxVoiceActivityDetectorFront(this.handle);

    let buffer =
        segment.deref()
            .samples.buffer.reinterpret(segment.deref().n * ref.sizeof.float)
            .buffer;

    let samples = new Float32Array(buffer).slice(0);
    let ans = new SpeechSegment(segment.deref().start, samples);

    libsherpa_onnx.SherpaOnnxDestroySpeechSegment(segment);
    return ans;
  }
};

class GeneratedAudio {
  constructor(sampleRate, samples) {
    this.sampleRate = sampleRate;
    this.samples = samples;
  }
  save(filename) {
    libsherpa_onnx.SherpaOnnxWriteWave(this.samples, this.samples.length,
                                       this.sampleRate, filename);
  }
};

class OfflineTts {
  constructor(config) {
    this.config = config;
    this.handle = libsherpa_onnx.SherpaOnnxCreateOfflineTts(config.ref());
  }

  free() {
    if (this.handle) {
      libsherpa_onnx.SherpaOnnxDestroyOfflineTts(this.handle);
      this.handle = null;
    }
  }
  generate(text, sid, speed) {
    let r = libsherpa_onnx.SherpaOnnxOfflineTtsGenerate(this.handle, text, sid,
                                                        speed);
    const buffer =
        r.deref()
            .samples.buffer.reinterpret(r.deref().n * ref.sizeof.float)
            .buffer;
    let samples = new Float32Array(buffer).slice(0);
    let sampleRate = r.deref().sampleRate;

    let generatedAudio = new GeneratedAudio(sampleRate, samples);

    libsherpa_onnx.SherpaOnnxDestroyOfflineTtsGeneratedAudio(r);

    return generatedAudio;
  }
};

// online asr
const OnlineTransducerModelConfig = SherpaOnnxOnlineTransducerModelConfig;
const OnlineModelConfig = SherpaOnnxOnlineModelConfig;
const FeatureConfig = SherpaOnnxFeatureConfig;
const OnlineRecognizerConfig = SherpaOnnxOnlineRecognizerConfig;
const OnlineParaformerModelConfig = SherpaOnnxOnlineParaformerModelConfig;
const OnlineZipformer2CtcModelConfig = SherpaOnnxOnlineZipformer2CtcModelConfig;

// offline asr
const OfflineTransducerModelConfig = SherpaOnnxOfflineTransducerModelConfig;
const OfflineModelConfig = SherpaOnnxOfflineModelConfig;
const OfflineRecognizerConfig = SherpaOnnxOfflineRecognizerConfig;
const OfflineParaformerModelConfig = SherpaOnnxOfflineParaformerModelConfig;
const OfflineWhisperModelConfig = SherpaOnnxOfflineWhisperModelConfig;
const OfflineNemoEncDecCtcModelConfig =
    SherpaOnnxOfflineNemoEncDecCtcModelConfig;
const OfflineTdnnModelConfig = SherpaOnnxOfflineTdnnModelConfig;

// vad
const SileroVadModelConfig = SherpaOnnxSileroVadModelConfig;
const VadModelConfig = SherpaOnnxVadModelConfig;

// tts
const OfflineTtsVitsModelConfig = SherpaOnnxOfflineTtsVitsModelConfig;
const OfflineTtsModelConfig = SherpaOnnxOfflineTtsModelConfig;
const OfflineTtsConfig = SherpaOnnxOfflineTtsConfig;

module.exports = {
  // online asr
  OnlineTransducerModelConfig,
  OnlineModelConfig,
  FeatureConfig,
  OnlineRecognizerConfig,
  OnlineRecognizer,
  OnlineStream,
  OnlineParaformerModelConfig,
  OnlineZipformer2CtcModelConfig,

  // offline asr
  OfflineRecognizer,
  OfflineStream,
  OfflineTransducerModelConfig,
  OfflineModelConfig,
  OfflineRecognizerConfig,
  OfflineParaformerModelConfig,
  OfflineWhisperModelConfig,
  OfflineNemoEncDecCtcModelConfig,
  OfflineTdnnModelConfig,
  // vad
  SileroVadModelConfig,
  VadModelConfig,
  CircularBuffer,
  VoiceActivityDetector,
  // tts
  OfflineTtsVitsModelConfig,
  OfflineTtsModelConfig,
  OfflineTtsConfig,
  OfflineTts,

  //
  Display,
};
