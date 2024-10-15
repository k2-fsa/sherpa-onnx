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
  modelType: String = "",
  modelingUnit: String = "cjkchar",
  bpeVocab: String = "",
  tokensBuf: String = "",
  tokensBufSize: Int = 0
) -> SherpaOnnxOnlineModelConfig {
  return SherpaOnnxOnlineModelConfig(
    transducer: transducer,
    paraformer: paraformer,
    zipformer2_ctc: zipformer2Ctc,
    tokens: toCPointer(tokens),
    num_threads: Int32(numThreads),
    provider: toCPointer(provider),
    debug: Int32(debug),
    model_type: toCPointer(modelType),
    modeling_unit: toCPointer(modelingUnit),
    bpe_vocab: toCPointer(bpeVocab),
    tokens_buf: toCPointer(tokensBuf),
    tokens_buf_size: Int32(tokensBufSize)
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

func sherpaOnnxOnlineCtcFstDecoderConfig(
  graph: String = "",
  maxActive: Int = 3000
) -> SherpaOnnxOnlineCtcFstDecoderConfig {
  return SherpaOnnxOnlineCtcFstDecoderConfig(
    graph: toCPointer(graph),
    max_active: Int32(maxActive))
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
  hotwordsScore: Float = 1.5,
  ctcFstDecoderConfig: SherpaOnnxOnlineCtcFstDecoderConfig = sherpaOnnxOnlineCtcFstDecoderConfig(),
  ruleFsts: String = "",
  ruleFars: String = "",
  blankPenalty: Float = 0.0,
  hotwordsBuf: String = "",
  hotwordsBufSize: Int = 0
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
    hotwords_score: hotwordsScore,
    ctc_fst_decoder_config: ctcFstDecoderConfig,
    rule_fsts: toCPointer(ruleFsts),
    rule_fars: toCPointer(ruleFars),
    blank_penalty: blankPenalty,
    hotwords_buf: toCPointer(hotwordsBuf),
    hotwords_buf_size: Int32(hotwordsBufSize)
  )
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

  init(result: UnsafePointer<SherpaOnnxOnlineRecognizerResult>!) {
    self.result = result
  }

  deinit {
    if let result {
      SherpaOnnxDestroyOnlineRecognizerResult(result)
    }
  }
}

class SherpaOnnxRecognizer {
  /// A pointer to the underlying counterpart in C
  let recognizer: OpaquePointer!
  var stream: OpaquePointer!

  /// Constructor taking a model config
  init(
    config: UnsafePointer<SherpaOnnxOnlineRecognizerConfig>!
  ) {
    recognizer = SherpaOnnxCreateOnlineRecognizer(config)
    stream = SherpaOnnxCreateOnlineStream(recognizer)
  }

  deinit {
    if let stream {
      SherpaOnnxDestroyOnlineStream(stream)
    }

    if let recognizer {
      SherpaOnnxDestroyOnlineRecognizer(recognizer)
    }
  }

  /// Decode wave samples.
  ///
  /// - Parameters:
  ///   - samples: Audio samples normalized to the range [-1, 1]
  ///   - sampleRate: Sample rate of the input audio samples. Must match
  ///                 the one expected by the model.
  func acceptWaveform(samples: [Float], sampleRate: Int = 16000) {
    SherpaOnnxOnlineStreamAcceptWaveform(stream, Int32(sampleRate), samples, Int32(samples.count))
  }

  func isReady() -> Bool {
    return SherpaOnnxIsOnlineStreamReady(recognizer, stream) == 1 ? true : false
  }

  /// If there are enough number of feature frames, it invokes the neural
  /// network computation and decoding. Otherwise, it is a no-op.
  func decode() {
    SherpaOnnxDecodeOnlineStream(recognizer, stream)
  }

  /// Get the decoding results so far
  func getResult() -> SherpaOnnxOnlineRecongitionResult {
    let result: UnsafePointer<SherpaOnnxOnlineRecognizerResult>? = SherpaOnnxGetOnlineStreamResult(
      recognizer, stream)
    return SherpaOnnxOnlineRecongitionResult(result: result)
  }

  /// Reset the recognizer, which clears the neural network model state
  /// and the state for decoding.
  /// If hotwords is an empty string, it just recreates the decoding stream
  /// If hotwords is not empty, it will create a new decoding stream with
  /// the given hotWords appended to the default hotwords.
  func reset(hotwords: String? = nil) {
    guard let words = hotwords, !words.isEmpty else {
      SherpaOnnxOnlineStreamReset(recognizer, stream)
      return
    }

    words.withCString { cString in
      let newStream = SherpaOnnxCreateOnlineStreamWithHotwords(recognizer, cString)
      // lock while release and replace stream
      objc_sync_enter(self)
      SherpaOnnxDestroyOnlineStream(stream)
      stream = newStream
      objc_sync_exit(self)
    }
  }

  /// Signal that no more audio samples would be available.
  /// After this call, you cannot call acceptWaveform() any more.
  func inputFinished() {
    SherpaOnnxOnlineStreamInputFinished(stream)
  }

  /// Return true is an endpoint has been detected.
  func isEndpoint() -> Bool {
    return SherpaOnnxOnlineStreamIsEndpoint(recognizer, stream) == 1 ? true : false
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
  decoder: String = "",
  language: String = "",
  task: String = "transcribe",
  tailPaddings: Int = -1
) -> SherpaOnnxOfflineWhisperModelConfig {
  return SherpaOnnxOfflineWhisperModelConfig(
    encoder: toCPointer(encoder),
    decoder: toCPointer(decoder),
    language: toCPointer(language),
    task: toCPointer(task),
    tail_paddings: Int32(tailPaddings)
  )
}

func sherpaOnnxOfflineTdnnModelConfig(
  model: String = ""
) -> SherpaOnnxOfflineTdnnModelConfig {
  return SherpaOnnxOfflineTdnnModelConfig(
    model: toCPointer(model)
  )
}

func sherpaOnnxOfflineSenseVoiceModelConfig(
  model: String = "",
  language: String = "",
  useInverseTextNormalization: Bool = false
) -> SherpaOnnxOfflineSenseVoiceModelConfig {
  return SherpaOnnxOfflineSenseVoiceModelConfig(
    model: toCPointer(model),
    language: toCPointer(language),
    use_itn: useInverseTextNormalization ? 1 : 0
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
  modelType: String = "",
  modelingUnit: String = "cjkchar",
  bpeVocab: String = "",
  teleSpeechCtc: String = "",
  senseVoice: SherpaOnnxOfflineSenseVoiceModelConfig = sherpaOnnxOfflineSenseVoiceModelConfig()
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
    model_type: toCPointer(modelType),
    modeling_unit: toCPointer(modelingUnit),
    bpe_vocab: toCPointer(bpeVocab),
    telespeech_ctc: toCPointer(teleSpeechCtc),
    sense_voice: senseVoice
  )
}

func sherpaOnnxOfflineRecognizerConfig(
  featConfig: SherpaOnnxFeatureConfig,
  modelConfig: SherpaOnnxOfflineModelConfig,
  lmConfig: SherpaOnnxOfflineLMConfig = sherpaOnnxOfflineLMConfig(),
  decodingMethod: String = "greedy_search",
  maxActivePaths: Int = 4,
  hotwordsFile: String = "",
  hotwordsScore: Float = 1.5,
  ruleFsts: String = "",
  ruleFars: String = "",
  blankPenalty: Float = 0.0
) -> SherpaOnnxOfflineRecognizerConfig {
  return SherpaOnnxOfflineRecognizerConfig(
    feat_config: featConfig,
    model_config: modelConfig,
    lm_config: lmConfig,
    decoding_method: toCPointer(decodingMethod),
    max_active_paths: Int32(maxActivePaths),
    hotwords_file: toCPointer(hotwordsFile),
    hotwords_score: hotwordsScore,
    rule_fsts: toCPointer(ruleFsts),
    rule_fars: toCPointer(ruleFars),
    blank_penalty: blankPenalty
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

  // For SenseVoice models, it can be zh, en, ja, yue, ko
  // where zh is for Chinese
  // en is for English
  // ja is for Japanese
  // yue is for Cantonese
  // ko is for Korean
  var lang: String {
    return String(cString: result.pointee.lang)
  }

  // for SenseVoice models
  var emotion: String {
    return String(cString: result.pointee.emotion)
  }

  // for SenseVoice models
  var event: String {
    return String(cString: result.pointee.event)
  }

  init(result: UnsafePointer<SherpaOnnxOfflineRecognizerResult>!) {
    self.result = result
  }

  deinit {
    if let result {
      SherpaOnnxDestroyOfflineRecognizerResult(result)
    }
  }
}

class SherpaOnnxOfflineRecognizer {
  /// A pointer to the underlying counterpart in C
  let recognizer: OpaquePointer!

  init(
    config: UnsafePointer<SherpaOnnxOfflineRecognizerConfig>!
  ) {
    recognizer = SherpaOnnxCreateOfflineRecognizer(config)
  }

  deinit {
    if let recognizer {
      SherpaOnnxDestroyOfflineRecognizer(recognizer)
    }
  }

  /// Decode wave samples.
  ///
  /// - Parameters:
  ///   - samples: Audio samples normalized to the range [-1, 1]
  ///   - sampleRate: Sample rate of the input audio samples. Must match
  ///                 the one expected by the model.
  func decode(samples: [Float], sampleRate: Int = 16000) -> SherpaOnnxOfflineRecongitionResult {
    let stream: OpaquePointer! = SherpaOnnxCreateOfflineStream(recognizer)

    SherpaOnnxAcceptWaveformOffline(stream, Int32(sampleRate), samples, Int32(samples.count))

    SherpaOnnxDecodeOfflineStream(recognizer, stream)

    let result: UnsafePointer<SherpaOnnxOfflineRecognizerResult>? =
      SherpaOnnxGetOfflineStreamResult(
        stream)

    SherpaOnnxDestroyOfflineStream(stream)

    return SherpaOnnxOfflineRecongitionResult(result: result)
  }
}

func sherpaOnnxSileroVadModelConfig(
  model: String,
  threshold: Float = 0.5,
  minSilenceDuration: Float = 0.25,
  minSpeechDuration: Float = 0.5,
  windowSize: Int = 512,
  maxSpeechDuration: Float = 5.0
) -> SherpaOnnxSileroVadModelConfig {
  return SherpaOnnxSileroVadModelConfig(
    model: toCPointer(model),
    threshold: threshold,
    min_silence_duration: minSilenceDuration,
    min_speech_duration: minSpeechDuration,
    window_size: Int32(windowSize),
    max_speech_duration: maxSpeechDuration
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

  func flush() {
    SherpaOnnxVoiceActivityDetectorFlush(vad)
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
  lengthScale: Float = 1.0,
  dictDir: String = ""
) -> SherpaOnnxOfflineTtsVitsModelConfig {
  return SherpaOnnxOfflineTtsVitsModelConfig(
    model: toCPointer(model),
    lexicon: toCPointer(lexicon),
    tokens: toCPointer(tokens),
    data_dir: toCPointer(dataDir),
    noise_scale: noiseScale,
    noise_scale_w: noiseScaleW,
    length_scale: lengthScale,
    dict_dir: toCPointer(dictDir))
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
  ruleFars: String = "",
  maxNumSentences: Int = 2
) -> SherpaOnnxOfflineTtsConfig {
  return SherpaOnnxOfflineTtsConfig(
    model: model,
    rule_fsts: toCPointer(ruleFsts),
    max_num_sentences: Int32(maxNumSentences),
    rule_fars: toCPointer(ruleFars)
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

typealias TtsCallbackWithArg = (
  @convention(c) (
    UnsafePointer<Float>?,  // const float* samples
    Int32,  // int32_t n
    UnsafeMutableRawPointer?  // void *arg
  ) -> Int32
)?

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

  func generateWithCallbackWithArg(
    text: String, callback: TtsCallbackWithArg, arg: UnsafeMutableRawPointer, sid: Int = 0,
    speed: Float = 1.0
  ) -> SherpaOnnxGeneratedAudioWrapper {
    let audio: UnsafePointer<SherpaOnnxGeneratedAudio>? =
      SherpaOnnxOfflineTtsGenerateWithCallbackWithArg(
        tts, toCPointer(text), Int32(sid), speed, callback, arg)

    return SherpaOnnxGeneratedAudioWrapper(audio: audio)
  }
}

// spoken language identification

func sherpaOnnxSpokenLanguageIdentificationWhisperConfig(
  encoder: String,
  decoder: String,
  tailPaddings: Int = -1
) -> SherpaOnnxSpokenLanguageIdentificationWhisperConfig {
  return SherpaOnnxSpokenLanguageIdentificationWhisperConfig(
    encoder: toCPointer(encoder),
    decoder: toCPointer(decoder),
    tail_paddings: Int32(tailPaddings))
}

func sherpaOnnxSpokenLanguageIdentificationConfig(
  whisper: SherpaOnnxSpokenLanguageIdentificationWhisperConfig,
  numThreads: Int = 1,
  debug: Int = 0,
  provider: String = "cpu"
) -> SherpaOnnxSpokenLanguageIdentificationConfig {
  return SherpaOnnxSpokenLanguageIdentificationConfig(
    whisper: whisper,
    num_threads: Int32(numThreads),
    debug: Int32(debug),
    provider: toCPointer(provider))
}

class SherpaOnnxSpokenLanguageIdentificationResultWrapper {
  /// A pointer to the underlying counterpart in C
  let result: UnsafePointer<SherpaOnnxSpokenLanguageIdentificationResult>!

  /// Return the detected language.
  /// en for English
  /// zh for Chinese
  /// es for Spanish
  /// de for German
  /// etc.
  var lang: String {
    return String(cString: result.pointee.lang)
  }

  init(result: UnsafePointer<SherpaOnnxSpokenLanguageIdentificationResult>!) {
    self.result = result
  }

  deinit {
    if let result {
      SherpaOnnxDestroySpokenLanguageIdentificationResult(result)
    }
  }
}

class SherpaOnnxSpokenLanguageIdentificationWrapper {
  /// A pointer to the underlying counterpart in C
  let slid: OpaquePointer!

  init(
    config: UnsafePointer<SherpaOnnxSpokenLanguageIdentificationConfig>!
  ) {
    slid = SherpaOnnxCreateSpokenLanguageIdentification(config)
  }

  deinit {
    if let slid {
      SherpaOnnxDestroySpokenLanguageIdentification(slid)
    }
  }

  func decode(samples: [Float], sampleRate: Int = 16000)
    -> SherpaOnnxSpokenLanguageIdentificationResultWrapper
  {
    let stream: OpaquePointer! = SherpaOnnxSpokenLanguageIdentificationCreateOfflineStream(slid)
    SherpaOnnxAcceptWaveformOffline(stream, Int32(sampleRate), samples, Int32(samples.count))

    let result: UnsafePointer<SherpaOnnxSpokenLanguageIdentificationResult>? =
      SherpaOnnxSpokenLanguageIdentificationCompute(
        slid,
        stream)

    SherpaOnnxDestroyOfflineStream(stream)
    return SherpaOnnxSpokenLanguageIdentificationResultWrapper(result: result)
  }
}

// keyword spotting

class SherpaOnnxKeywordResultWrapper {
  /// A pointer to the underlying counterpart in C
  let result: UnsafePointer<SherpaOnnxKeywordResult>!

  var keyword: String {
    return String(cString: result.pointee.keyword)
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

  init(result: UnsafePointer<SherpaOnnxKeywordResult>!) {
    self.result = result
  }

  deinit {
    if let result {
      SherpaOnnxDestroyKeywordResult(result)
    }
  }
}

func sherpaOnnxKeywordSpotterConfig(
  featConfig: SherpaOnnxFeatureConfig,
  modelConfig: SherpaOnnxOnlineModelConfig,
  keywordsFile: String,
  maxActivePaths: Int = 4,
  numTrailingBlanks: Int = 1,
  keywordsScore: Float = 1.0,
  keywordsThreshold: Float = 0.25,
  keywordsBuf: String = "",
  keywordsBufSize: Int = 0
) -> SherpaOnnxKeywordSpotterConfig {
  return SherpaOnnxKeywordSpotterConfig(
    feat_config: featConfig,
    model_config: modelConfig,
    max_active_paths: Int32(maxActivePaths),
    num_trailing_blanks: Int32(numTrailingBlanks),
    keywords_score: keywordsScore,
    keywords_threshold: keywordsThreshold,
    keywords_file: toCPointer(keywordsFile),
    keywords_buf: toCPointer(keywordsBuf),
    keywords_buf_size: Int32(keywordsBufSize)
  )
}

class SherpaOnnxKeywordSpotterWrapper {
  /// A pointer to the underlying counterpart in C
  let spotter: OpaquePointer!
  var stream: OpaquePointer!

  init(
    config: UnsafePointer<SherpaOnnxKeywordSpotterConfig>!
  ) {
    spotter = SherpaOnnxCreateKeywordSpotter(config)
    stream = SherpaOnnxCreateKeywordStream(spotter)
  }

  deinit {
    if let stream {
      SherpaOnnxDestroyOnlineStream(stream)
    }

    if let spotter {
      SherpaOnnxDestroyKeywordSpotter(spotter)
    }
  }

  func acceptWaveform(samples: [Float], sampleRate: Int = 16000) {
    SherpaOnnxOnlineStreamAcceptWaveform(stream, Int32(sampleRate), samples, Int32(samples.count))
  }

  func isReady() -> Bool {
    return SherpaOnnxIsKeywordStreamReady(spotter, stream) == 1 ? true : false
  }

  func decode() {
    SherpaOnnxDecodeKeywordStream(spotter, stream)
  }

  func getResult() -> SherpaOnnxKeywordResultWrapper {
    let result: UnsafePointer<SherpaOnnxKeywordResult>? = SherpaOnnxGetKeywordResult(
      spotter, stream)
    return SherpaOnnxKeywordResultWrapper(result: result)
  }

  /// Signal that no more audio samples would be available.
  /// After this call, you cannot call acceptWaveform() any more.
  func inputFinished() {
    SherpaOnnxOnlineStreamInputFinished(stream)
  }
}

// Punctuation

func sherpaOnnxOfflinePunctuationModelConfig(
  ctTransformer: String,
  numThreads: Int = 1,
  debug: Int = 0,
  provider: String = "cpu"
) -> SherpaOnnxOfflinePunctuationModelConfig {
  return SherpaOnnxOfflinePunctuationModelConfig(
    ct_transformer: toCPointer(ctTransformer),
    num_threads: Int32(numThreads),
    debug: Int32(debug),
    provider: toCPointer(provider)
  )
}

func sherpaOnnxOfflinePunctuationConfig(
  model: SherpaOnnxOfflinePunctuationModelConfig
) -> SherpaOnnxOfflinePunctuationConfig {
  return SherpaOnnxOfflinePunctuationConfig(
    model: model
  )
}

class SherpaOnnxOfflinePunctuationWrapper {
  /// A pointer to the underlying counterpart in C
  let ptr: OpaquePointer!

  /// Constructor taking a model config
  init(
    config: UnsafePointer<SherpaOnnxOfflinePunctuationConfig>!
  ) {
    ptr = SherpaOnnxCreateOfflinePunctuation(config)
  }

  deinit {
    if let ptr {
      SherpaOnnxDestroyOfflinePunctuation(ptr)
    }
  }

  func addPunct(text: String) -> String {
    let cText = SherpaOfflinePunctuationAddPunct(ptr, toCPointer(text))
    let ans = String(cString: cText!)
    SherpaOfflinePunctuationFreeText(cText)
    return ans
  }
}

func sherpaOnnxOfflineSpeakerSegmentationPyannoteModelConfig(model: String)
  -> SherpaOnnxOfflineSpeakerSegmentationPyannoteModelConfig
{
  return SherpaOnnxOfflineSpeakerSegmentationPyannoteModelConfig(model: toCPointer(model))
}

func sherpaOnnxOfflineSpeakerSegmentationModelConfig(
  pyannote: SherpaOnnxOfflineSpeakerSegmentationPyannoteModelConfig,
  numThreads: Int = 1,
  debug: Int = 0,
  provider: String = "cpu"
) -> SherpaOnnxOfflineSpeakerSegmentationModelConfig {
  return SherpaOnnxOfflineSpeakerSegmentationModelConfig(
    pyannote: pyannote,
    num_threads: Int32(numThreads),
    debug: Int32(debug),
    provider: toCPointer(provider)
  )
}

func sherpaOnnxFastClusteringConfig(numClusters: Int = -1, threshold: Float = 0.5)
  -> SherpaOnnxFastClusteringConfig
{
  return SherpaOnnxFastClusteringConfig(num_clusters: Int32(numClusters), threshold: threshold)
}

func sherpaOnnxSpeakerEmbeddingExtractorConfig(
  model: String,
  numThreads: Int = 1,
  debug: Int = 0,
  provider: String = "cpu"
) -> SherpaOnnxSpeakerEmbeddingExtractorConfig {
  return SherpaOnnxSpeakerEmbeddingExtractorConfig(
    model: toCPointer(model),
    num_threads: Int32(numThreads),
    debug: Int32(debug),
    provider: toCPointer(provider)
  )
}

func sherpaOnnxOfflineSpeakerDiarizationConfig(
  segmentation: SherpaOnnxOfflineSpeakerSegmentationModelConfig,
  embedding: SherpaOnnxSpeakerEmbeddingExtractorConfig,
  clustering: SherpaOnnxFastClusteringConfig,
  minDurationOn: Float = 0.3,
  minDurationOff: Float = 0.5
) -> SherpaOnnxOfflineSpeakerDiarizationConfig {
  return SherpaOnnxOfflineSpeakerDiarizationConfig(
    segmentation: segmentation,
    embedding: embedding,
    clustering: clustering,
    min_duration_on: minDurationOn,
    min_duration_off: minDurationOff
  )
}

struct SherpaOnnxOfflineSpeakerDiarizationSegmentWrapper {
  var start: Float = 0
  var end: Float = 0
  var speaker: Int = 0
}

class SherpaOnnxOfflineSpeakerDiarizationWrapper {
  /// A pointer to the underlying counterpart in C
  let impl: OpaquePointer!

  init(
    config: UnsafePointer<SherpaOnnxOfflineSpeakerDiarizationConfig>!
  ) {
    impl = SherpaOnnxCreateOfflineSpeakerDiarization(config)
  }

  deinit {
    if let impl {
      SherpaOnnxDestroyOfflineSpeakerDiarization(impl)
    }
  }

  var sampleRate: Int {
    return Int(SherpaOnnxOfflineSpeakerDiarizationGetSampleRate(impl))
  }

  // only config.clustering is used. All other fields are ignored
  func setConfig(config: UnsafePointer<SherpaOnnxOfflineSpeakerDiarizationConfig>!) {
    SherpaOnnxOfflineSpeakerDiarizationSetConfig(impl, config)
  }

  func process(samples: [Float]) -> [SherpaOnnxOfflineSpeakerDiarizationSegmentWrapper] {
    let result = SherpaOnnxOfflineSpeakerDiarizationProcess(
      impl, samples, Int32(samples.count))

    if result == nil {
      return []
    }

    let numSegments = Int(SherpaOnnxOfflineSpeakerDiarizationResultGetNumSegments(result))

    let p: UnsafePointer<SherpaOnnxOfflineSpeakerDiarizationSegment>? =
      SherpaOnnxOfflineSpeakerDiarizationResultSortByStartTime(result)

    if p == nil {
      return []
    }

    var ans: [SherpaOnnxOfflineSpeakerDiarizationSegmentWrapper] = []
    for i in 0..<numSegments {
      ans.append(
        SherpaOnnxOfflineSpeakerDiarizationSegmentWrapper(
          start: p![i].start, end: p![i].end, speaker: Int(p![i].speaker)))
    }

    SherpaOnnxOfflineSpeakerDiarizationDestroySegment(p)
    SherpaOnnxOfflineSpeakerDiarizationDestroyResult(result)

    return ans
  }
}
