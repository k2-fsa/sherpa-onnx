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
/// https://k2-fsa.github.io/sherpa/onnx/pretrained_models/index.html
/// to download the required `.onnx` files.
///
/// - Parameters:
///   - encoder: Path to encoder.onnx
///   - decoder: Path to decoder.onnx
///   - joiner: Path to joiner.onnx
///   - tokens: Path to tokens.txt
///   - numThreads:  Number of threads to use for neural network computation.
///
/// - Returns: Return an instance of SherpaOnnxOnlineTransducerModelConfig
func sherpaOnnxOnlineTransducerModelConfig(
  encoder: String,
  decoder: String,
  joiner: String,
  tokens: String,
  numThreads: Int = 2,
  debug: Int = 0
) -> SherpaOnnxOnlineTransducerModelConfig{
  return SherpaOnnxOnlineTransducerModelConfig(
    encoder: toCPointer(encoder),
    decoder: toCPointer(decoder),
    joiner: toCPointer(joiner),
    tokens: toCPointer(tokens),
    num_threads: Int32(numThreads),
    debug: Int32(debug)
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
    modelConfig: SherpaOnnxOnlineTransducerModelConfig,
    enableEndpoint: Bool = false,
    rule1MinTrailingSilence: Float = 2.4,
    rule2MinTrailingSilence: Float = 1.2,
    rule3MinUtteranceLength: Float = 30,
    decodingMethod: String = "greedy_search",
    maxActivePaths: Int = 4
) ->  SherpaOnnxOnlineRecognizerConfig{
  return SherpaOnnxOnlineRecognizerConfig(
    feat_config: featConfig,
    model_config: modelConfig,
    decoding_method: toCPointer(decodingMethod),
    max_active_paths: Int32(maxActivePaths),
    enable_endpoint: enableEndpoint ? 1 : 0,
    rule1_min_trailing_silence: rule1MinTrailingSilence,
    rule2_min_trailing_silence: rule2MinTrailingSilence,
    rule3_min_utterance_length: rule3MinUtteranceLength)
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

  /// Constructor taking a model config and a decoder config.
  init(
    config: UnsafePointer<SherpaOnnxOnlineRecognizerConfig>!
  ) {
    recognizer = CreateOnlineRecognizer(config)
    stream = CreateOnlineStream(recognizer)
  }

  deinit {
    if let stream {
      DestoryOnlineStream(stream)
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
  ///                 the one expected by the model. It must be 16000 for
  ///                 models from icefall.
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
    let result: UnsafeMutablePointer<SherpaOnnxOnlineRecognizerResult>? = GetOnlineStreamResult(recognizer, stream)
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
