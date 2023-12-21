/// swfit-api-examples/SherpaOnnx.swift
/// Copyright (c)  2023  Xiaomi Corporation

import Foundation  // For NSString

/// Convert a String from swift to a `const char*` so that we can pass it to
/// the C language.
///
/// - Parameters:
///   - s: The String to convert.
/// - Returns: A pointer that can be passed to C as `const char*`

func toCPointer(_ s: String) -> UnsafePointer<Int8>! {
  let cs = (s as NSString).utf8String
  return UnsafePointer<Int8>(cs)
}

/// Return an instance of SherpaOnnxOnlineTransducerModelConfig.
///
/// Please refer to
/// https://k2-fsa.github.io/sherpa/onnx/pretrained_models/online-transducer/index.html
/// to download the required `.onnx` files.
///
/// - Parameters:
///   - encoder: Path to encoder.onnx
///   - decoder: Path to decoder.onnx
///   - joiner: Path to joiner.onnx
///
/// - Returns: Return an instance of SherpaOnnxOnlineTransducerModelConfig
func sherpaOnnxOnlineTransducerModelConfig(
  encoder: String = "",
  decoder: String = "",
  joiner: String = ""
) -> SherpaOnnxOnlineTransducerModelConfig {
  return SherpaOnnxOnlineTransducerModelConfig(
    encoder: toCPointer(encoder),
    decoder: toCPointer(decoder),
    joiner: toCPointer(joiner)
  )
}

/// Return an instance of SherpaOnnxOnlineParaformerModelConfig.
///
/// Please refer to
/// https://k2-fsa.github.io/sherpa/onnx/pretrained_models/online-paraformer/index.html
/// to download the required `.onnx` files.
///
/// - Parameters:
///   - encoder: Path to encoder.onnx
///   - decoder: Path to decoder.onnx
///
/// - Returns: Return an instance of SherpaOnnxOnlineParaformerModelConfig
func sherpaOnnxOnlineParaformerModelConfig(
  encoder: String = "",
  decoder: String = ""
) -> SherpaOnnxOnlineParaformerModelConfig {
  return SherpaOnnxOnlineParaformerModelConfig(
    encoder: toCPointer(encoder),
    decoder: toCPointer(decoder)
  )
}

func sherpaOnnxOnlineZipformer2CtcModelConfig(
  model: String = ""
) -> SherpaOnnxOnlineZipformer2CtcModelConfig {
  return SherpaOnnxOnlineZipformer2CtcModelConfig(
    model: toCPointer(model)
  )
}

/// Return an instance of SherpaOnnxOnlineModelConfig.
///
/// Please refer to
/// https://k2-fsa.github.io/sherpa/onnx/pretrained_models/index.html
/// to download the required `.onnx` files.
///
/// - Parameters:
///   - tokens: Path to tokens.txt
///   - numThreads:  Number of threads to use for neural network computation.
///
/// - Returns: Return an instance of SherpaOnnxOnlineTransducerModelConfig
func sherpaOnnxOnlineModelConfig(
  tokens: String,
  transducer: SherpaOnnxOnlineTransducerModelConfig = sherpaOnnxOnlineTransducerModelConfig(),
  paraformer: SherpaOnnxOnlineParaformerModelConfig = sherpaOnnxOnlineParaformerModelConfig(),
  zipformer2Ctc: SherpaOnnxOnlineZipformer2CtcModelConfig =
    sherpaOnnxOnlineZipformer2CtcModelConfig(),
  numThreads: Int = 1,
  provider: String = "cpu",
  debug: Int = 0,
  modelType: String = ""
) -> SherpaOnnxOnlineModelConfig {
  return SherpaOnnxOnlineModelConfig(
    transducer: transducer,
    paraformer: paraformer,
    zipformer2_ctc: zipformer2Ctc,
    tokens: toCPointer(tokens),
    num_threads: Int32(numThreads),
    provider: toCPointer(provider),
    debug: Int32(debug),
    model_type: toCPointer(modelType)
  )
}

func sherpaOnnxFeatureConfig(
  sampleRate: Int = 16000,
  featureDim: Int = 80
) -> SherpaOnnxFeatureConfig {
  return SherpaOnnxFeatureConfig(
    sample_rate: Int32(sampleRate),
    feature_dim: Int32(featureDim))
}

func sherpaOnnxOnlineRecognizerConfig(
  featConfig: SherpaOnnxFeatureConfig,
  modelConfig: SherpaOnnxOnlineModelConfig,
  enableEndpoint: Bool = false,
  rule1MinTrailingSilence: Float = 2.4,
  rule2MinTrailingSilence: Float = 1.2,
  rule3MinUtteranceLength: Float = 30,
  decodingMethod: String = "greedy_search",
  maxActivePaths: Int = 4,
  hotwordsFile: String = "",
  hotwordsScore: Float = 1.5
) -> SherpaOnnxOnlineRecognizerConfig {
  return SherpaOnnxOnlineRecognizerConfig(
    feat_config: featConfig,
    model_config: modelConfig,
    decoding_method: toCPointer(decodingMethod),
    max_active_paths: Int32(maxActivePaths),
    enable_endpoint: enableEndpoint ? 1 : 0,
    rule1_min_trailing_silence: rule1MinTrailingSilence,
    rule2_min_trailing_silence: rule2MinTrailingSilence,
    rule3_min_utterance_length: rule3MinUtteranceLength,
    hotwords_file: toCPointer(hotwordsFile),
    hotwords_score: hotwordsScore)
}

/// Wrapper for recognition result.
///
/// Usage:
///
///  let result = recognizer.getResult()
///  print("text: \(result.text)")
///
class SherpaOnnxOnlineRecongitionResult {
  /// A pointer to the underlying counterpart in C
  let result: UnsafePointer<SherpaOnnxOnlineRecognizerResult>!

  /// Return the actual recognition result.
  /// For English models, it contains words separated by spaces.
  /// For Chinese models, it contains Chinese words.
  var text: String {
    return String(cString: result.pointee.text)
  }

  var count: Int32 {
    return result.pointee.count
  }

  var tokens: [String] {
    if let tokensPointer = result.pointee.tokens_arr {
      var tokens: [String] = []
      for index in 0..<count {
        if let tokenPointer = tokensPointer[Int(index)] {
          let token = String(cString: tokenPointer)
          tokens.append(token)
        }
      }
      return tokens
    } else {
      let tokens: [String] = []
      return tokens
    }
  }

  init(result: UnsafePointer<SherpaOnnxOnlineRecognizerResult>!) {
    self.result = result
  }

  deinit {
    if let result {
      DestroyOnlineRecognizerResult(result)
    }
  }
}

class SherpaOnnxRecognizer {
  /// A pointer to the underlying counterpart in C
  let recognizer: OpaquePointer!
  let stream: OpaquePointer!

  /// Constructor taking a model config
  init(
    config: UnsafePointer<SherpaOnnxOnlineRecognizerConfig>!
  ) {
    recognizer = CreateOnlineRecognizer(config)
    stream = CreateOnlineStream(recognizer)
  }

  deinit {
    if let stream {
      DestroyOnlineStream(stream)
    }

    if let recognizer {
      DestroyOnlineRecognizer(recognizer)
    }
  }

  /// Decode wave samples.
  ///
  /// - Parameters:
  ///   - samples: Audio samples normalized to the range [-1, 1]
  ///   - sampleRate: Sample rate of the input audio samples. Must match
  ///                 the one expected by the model.
  func acceptWaveform(samples: [Float], sampleRate: Int = 16000) {
    AcceptWaveform(stream, Int32(sampleRate), samples, Int32(samples.count))
  }

  func isReady() -> Bool {
    return IsOnlineStreamReady(recognizer, stream) == 1 ? true : false
  }

  /// If there are enough number of feature frames, it invokes the neural
  /// network computation and decoding. Otherwise, it is a no-op.
  func decode() {
    DecodeOnlineStream(recognizer, stream)
  }

  /// Get the decoding results so far
  func getResult() -> SherpaOnnxOnlineRecongitionResult {
    let result: UnsafePointer<SherpaOnnxOnlineRecognizerResult>? = GetOnlineStreamResult(
      recognizer, stream)
    return SherpaOnnxOnlineRecongitionResult(result: result)
  }

  /// Reset the recognizer, which clears the neural network model state
  /// and the state for decoding.
  func reset() {
    Reset(recognizer, stream)
  }

  /// Signal that no more audio samples would be available.
  /// After this call, you cannot call acceptWaveform() any more.
  func inputFinished() {
    InputFinished(stream)
  }

  /// Return true is an endpoint has been detected.
  func isEndpoint() -> Bool {
    return IsEndpoint(recognizer, stream) == 1 ? true : false
  }
}

// For offline APIs

func sherpaOnnxOfflineTransducerModelConfig(
  encoder: String = "",
  decoder: String = "",
  joiner: String = ""
) -> SherpaOnnxOfflineTransducerModelConfig {
  return SherpaOnnxOfflineTransducerModelConfig(
    encoder: toCPointer(encoder),
    decoder: toCPointer(decoder),
    joiner: toCPointer(joiner)
  )
}

func sherpaOnnxOfflineParaformerModelConfig(
  model: String = ""
) -> SherpaOnnxOfflineParaformerModelConfig {
  return SherpaOnnxOfflineParaformerModelConfig(
    model: toCPointer(model)
  )
}

func sherpaOnnxOfflineNemoEncDecCtcModelConfig(
  model: String = ""
) -> SherpaOnnxOfflineNemoEncDecCtcModelConfig {
  return SherpaOnnxOfflineNemoEncDecCtcModelConfig(
    model: toCPointer(model)
  )
}

func sherpaOnnxOfflineWhisperModelConfig(
  encoder: String = "",
  decoder: String = ""
) -> SherpaOnnxOfflineWhisperModelConfig {
  return SherpaOnnxOfflineWhisperModelConfig(
    encoder: toCPointer(encoder),
    decoder: toCPointer(decoder)
  )
}

func sherpaOnnxOfflineTdnnModelConfig(
  model: String = ""
) -> SherpaOnnxOfflineTdnnModelConfig {
  return SherpaOnnxOfflineTdnnModelConfig(
    model: toCPointer(model)
  )
}

func sherpaOnnxOfflineLMConfig(
  model: String = "",
  scale: Float = 1.0
) -> SherpaOnnxOfflineLMConfig {
  return SherpaOnnxOfflineLMConfig(
    model: toCPointer(model),
    scale: scale
  )
}

func sherpaOnnxOfflineModelConfig(
  tokens: String,
  transducer: SherpaOnnxOfflineTransducerModelConfig = sherpaOnnxOfflineTransducerModelConfig(),
  paraformer: SherpaOnnxOfflineParaformerModelConfig = sherpaOnnxOfflineParaformerModelConfig(),
  nemoCtc: SherpaOnnxOfflineNemoEncDecCtcModelConfig = sherpaOnnxOfflineNemoEncDecCtcModelConfig(),
  whisper: SherpaOnnxOfflineWhisperModelConfig = sherpaOnnxOfflineWhisperModelConfig(),
  tdnn: SherpaOnnxOfflineTdnnModelConfig = sherpaOnnxOfflineTdnnModelConfig(),
  numThreads: Int = 1,
  provider: String = "cpu",
  debug: Int = 0,
  modelType: String = ""
) -> SherpaOnnxOfflineModelConfig {
  return SherpaOnnxOfflineModelConfig(
    transducer: transducer,
    paraformer: paraformer,
    nemo_ctc: nemoCtc,
    whisper: whisper,
    tdnn: tdnn,
    tokens: toCPointer(tokens),
    num_threads: Int32(numThreads),
    debug: Int32(debug),
    provider: toCPointer(provider),
    model_type: toCPointer(modelType)
  )
}

func sherpaOnnxOfflineRecognizerConfig(
  featConfig: SherpaOnnxFeatureConfig,
  modelConfig: SherpaOnnxOfflineModelConfig,
  lmConfig: SherpaOnnxOfflineLMConfig = sherpaOnnxOfflineLMConfig(),
  decodingMethod: String = "greedy_search",
  maxActivePaths: Int = 4,
  hotwordsFile: String = "",
  hotwordsScore: Float = 1.5
) -> SherpaOnnxOfflineRecognizerConfig {
  return SherpaOnnxOfflineRecognizerConfig(
    feat_config: featConfig,
    model_config: modelConfig,
    lm_config: lmConfig,
    decoding_method: toCPointer(decodingMethod),
    max_active_paths: Int32(maxActivePaths),
    hotwords_file: toCPointer(hotwordsFile),
    hotwords_score: hotwordsScore
  )
}

class SherpaOnnxOfflineRecongitionResult {
  /// A pointer to the underlying counterpart in C
  let result: UnsafePointer<SherpaOnnxOfflineRecognizerResult>!

  /// Return the actual recognition result.
  /// For English models, it contains words separated by spaces.
  /// For Chinese models, it contains Chinese words.
  var text: String {
    return String(cString: result.pointee.text)
  }

  var count: Int32 {
    return result.pointee.count
  }

  var timestamps: [Float] {
    if let p = result.pointee.timestamps {
      var timestamps: [Float] = []
      for index in 0..<count {
        timestamps.append(p[Int(index)])
      }
      return timestamps
    } else {
      let timestamps: [Float] = []
      return timestamps
    }
  }

  init(result: UnsafePointer<SherpaOnnxOfflineRecognizerResult>!) {
    self.result = result
  }

  deinit {
    if let result {
      DestroyOfflineRecognizerResult(result)
    }
  }
}

class SherpaOnnxOfflineRecognizer {
  /// A pointer to the underlying counterpart in C
  let recognizer: OpaquePointer!

  init(
    config: UnsafePointer<SherpaOnnxOfflineRecognizerConfig>!
  ) {
    recognizer = CreateOfflineRecognizer(config)
  }

  deinit {
    if let recognizer {
      DestroyOfflineRecognizer(recognizer)
    }
  }

  /// Decode wave samples.
  ///
  /// - Parameters:
  ///   - samples: Audio samples normalized to the range [-1, 1]
  ///   - sampleRate: Sample rate of the input audio samples. Must match
  ///                 the one expected by the model.
  func decode(samples: [Float], sampleRate: Int = 16000) -> SherpaOnnxOfflineRecongitionResult {
    let stream: OpaquePointer! = CreateOfflineStream(recognizer)

    AcceptWaveformOffline(stream, Int32(sampleRate), samples, Int32(samples.count))

    DecodeOfflineStream(recognizer, stream)

    let result: UnsafePointer<SherpaOnnxOfflineRecognizerResult>? = GetOfflineStreamResult(
      stream)

    DestroyOfflineStream(stream)

    return SherpaOnnxOfflineRecongitionResult(result: result)
  }
}

func sherpaOnnxSileroVadModelConfig(
  model: String,
  threshold: Float = 0.5,
  minSilenceDuration: Float = 0.25,
  minSpeechDuration: Float = 0.5,
  windowSize: Int = 512
) -> SherpaOnnxSileroVadModelConfig {
  return SherpaOnnxSileroVadModelConfig(
    model: toCPointer(model),
    threshold: threshold,
    min_silence_duration: minSilenceDuration,
    min_speech_duration: minSpeechDuration,
    window_size: Int32(windowSize)
  )
}

func sherpaOnnxVadModelConfig(
  sileroVad: SherpaOnnxSileroVadModelConfig,
  sampleRate: Int32 = 16000,
  numThreads: Int = 1,
  provider: String = "cpu",
  debug: Int = 0
) -> SherpaOnnxVadModelConfig {
  return SherpaOnnxVadModelConfig(
    silero_vad: sileroVad,
    sample_rate: sampleRate,
    num_threads: Int32(numThreads),
    provider: toCPointer(provider),
    debug: Int32(debug)
  )
}

class SherpaOnnxCircularBufferWrapper {
  let buffer: OpaquePointer!

  init(capacity: Int) {
    buffer = SherpaOnnxCreateCircularBuffer(Int32(capacity))
  }

  deinit {
    if let buffer {
      SherpaOnnxDestroyCircularBuffer(buffer)
    }
  }

  func push(samples: [Float]) {
    SherpaOnnxCircularBufferPush(buffer, samples, Int32(samples.count))
  }

  func get(startIndex: Int, n: Int) -> [Float] {
    let p: UnsafePointer<Float>! = SherpaOnnxCircularBufferGet(buffer, Int32(startIndex), Int32(n))

    var samples: [Float] = []

    for index in 0..<n {
      samples.append(p[Int(index)])
    }

    SherpaOnnxCircularBufferFree(p)

    return samples
  }

  func pop(n: Int) {
    SherpaOnnxCircularBufferPop(buffer, Int32(n))
  }

  func size() -> Int {
    return Int(SherpaOnnxCircularBufferSize(buffer))
  }

  func reset() {
    SherpaOnnxCircularBufferReset(buffer)
  }
}

class SherpaOnnxSpeechSegmentWrapper {
  let p: UnsafePointer<SherpaOnnxSpeechSegment>!

  init(p: UnsafePointer<SherpaOnnxSpeechSegment>!) {
    self.p = p
  }

  deinit {
    if let p {
      SherpaOnnxDestroySpeechSegment(p)
    }
  }

  var start: Int {
    return Int(p.pointee.start)
  }

  var n: Int {
    return Int(p.pointee.n)
  }

  var samples: [Float] {
    var samples: [Float] = []
    for index in 0..<n {
      samples.append(p.pointee.samples[Int(index)])
    }
    return samples
  }
}

class SherpaOnnxVoiceActivityDetectorWrapper {
  /// A pointer to the underlying counterpart in C
  let vad: OpaquePointer!

  init(config: UnsafePointer<SherpaOnnxVadModelConfig>!, buffer_size_in_seconds: Float) {
    vad = SherpaOnnxCreateVoiceActivityDetector(config, buffer_size_in_seconds)
  }

  deinit {
    if let vad {
      SherpaOnnxDestroyVoiceActivityDetector(vad)
    }
  }

  func acceptWaveform(samples: [Float]) {
    SherpaOnnxVoiceActivityDetectorAcceptWaveform(vad, samples, Int32(samples.count))
  }

  func isEmpty() -> Bool {
    return SherpaOnnxVoiceActivityDetectorEmpty(vad) == 1
  }

  func isSpeechDetected() -> Bool {
    return SherpaOnnxVoiceActivityDetectorDetected(vad) == 1
  }

  func pop() {
    SherpaOnnxVoiceActivityDetectorPop(vad)
  }

  func clear() {
    SherpaOnnxVoiceActivityDetectorClear(vad)
  }

  func front() -> SherpaOnnxSpeechSegmentWrapper {
    let p: UnsafePointer<SherpaOnnxSpeechSegment>? = SherpaOnnxVoiceActivityDetectorFront(vad)
    return SherpaOnnxSpeechSegmentWrapper(p: p)
  }

  func reset() {
    SherpaOnnxVoiceActivityDetectorReset(vad)
  }
}

// offline tts
func sherpaOnnxOfflineTtsVitsModelConfig(
  model: String,
  lexicon: String,
  tokens: String,
  dataDir: String = "",
  noiseScale: Float = 0.667,
  noiseScaleW: Float = 0.8,
  lengthScale: Float = 1.0
) -> SherpaOnnxOfflineTtsVitsModelConfig {
  return SherpaOnnxOfflineTtsVitsModelConfig(
    model: toCPointer(model),
    lexicon: toCPointer(lexicon),
    tokens: toCPointer(tokens),
    data_dir: toCPointer(dataDir),
    noise_scale: noiseScale,
    noise_scale_w: noiseScaleW,
    length_scale: lengthScale)
}

func sherpaOnnxOfflineTtsModelConfig(
  vits: SherpaOnnxOfflineTtsVitsModelConfig,
  numThreads: Int = 1,
  debug: Int = 0,
  provider: String = "cpu"
) -> SherpaOnnxOfflineTtsModelConfig {
  return SherpaOnnxOfflineTtsModelConfig(
    vits: vits,
    num_threads: Int32(numThreads),
    debug: Int32(debug),
    provider: toCPointer(provider)
  )
}

func sherpaOnnxOfflineTtsConfig(
  model: SherpaOnnxOfflineTtsModelConfig,
  ruleFsts: String = "",
  maxNumSenetences: Int = 2
) -> SherpaOnnxOfflineTtsConfig {
  return SherpaOnnxOfflineTtsConfig(
    model: model,
    rule_fsts: toCPointer(ruleFsts),
    max_num_sentences: Int32(maxNumSenetences)
  )
}

class SherpaOnnxGeneratedAudioWrapper {
  /// A pointer to the underlying counterpart in C
  let audio: UnsafePointer<SherpaOnnxGeneratedAudio>!

  init(audio: UnsafePointer<SherpaOnnxGeneratedAudio>!) {
    self.audio = audio
  }

  deinit {
    if let audio {
      SherpaOnnxDestroyOfflineTtsGeneratedAudio(audio)
    }
  }

  var n: Int32 {
    return audio.pointee.n
  }

  var sampleRate: Int32 {
    return audio.pointee.sample_rate
  }

  var samples: [Float] {
    if let p = audio.pointee.samples {
      var samples: [Float] = []
      for index in 0..<n {
        samples.append(p[Int(index)])
      }
      return samples
    } else {
      let samples: [Float] = []
      return samples
    }
  }

  func save(filename: String) -> Int32 {
    return SherpaOnnxWriteWave(audio.pointee.samples, n, sampleRate, toCPointer(filename))
  }
}

class SherpaOnnxOfflineTtsWrapper {
  /// A pointer to the underlying counterpart in C
  let tts: OpaquePointer!

  /// Constructor taking a model config
  init(
    config: UnsafePointer<SherpaOnnxOfflineTtsConfig>!
  ) {
    tts = SherpaOnnxCreateOfflineTts(config)
  }

  deinit {
    if let tts {
      SherpaOnnxDestroyOfflineTts(tts)
    }
  }

  func generate(text: String, sid: Int = 0, speed: Float = 1.0) -> SherpaOnnxGeneratedAudioWrapper {
    let audio: UnsafePointer<SherpaOnnxGeneratedAudio>? = SherpaOnnxOfflineTtsGenerate(
      tts, toCPointer(text), Int32(sid), speed)

    return SherpaOnnxGeneratedAudioWrapper(audio: audio)
  }
}
