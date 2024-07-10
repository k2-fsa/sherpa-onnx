import AVFoundation

extension AudioBuffer {
  func array() -> [Float] {
    return Array(UnsafeBufferPointer(self))
  }
}

extension AVAudioPCMBuffer {
  func array() -> [Float] {
    return self.audioBufferList.pointee.mBuffers.array()
  }
}

func run() {
  let filePath = "./sherpa-onnx-kws-zipformer-wenetspeech-3.3M-2024-01-01/test_wavs/3.wav"
  let encoder =
    "./sherpa-onnx-kws-zipformer-wenetspeech-3.3M-2024-01-01/encoder-epoch-12-avg-2-chunk-16-left-64.onnx"
  let decoder =
    "./sherpa-onnx-kws-zipformer-wenetspeech-3.3M-2024-01-01/decoder-epoch-12-avg-2-chunk-16-left-64.onnx"
  let joiner =
    "./sherpa-onnx-kws-zipformer-wenetspeech-3.3M-2024-01-01/joiner-epoch-12-avg-2-chunk-16-left-64.onnx"
  let tokens =
    "./sherpa-onnx-kws-zipformer-wenetspeech-3.3M-2024-01-01/tokens.txt"
  let keywordsFile =
    "./sherpa-onnx-kws-zipformer-wenetspeech-3.3M-2024-01-01/test_wavs/test_keywords.txt"
  let transducerConfig = sherpaOnnxOnlineTransducerModelConfig(
    encoder: encoder,
    decoder: decoder,
    joiner: joiner
  )

  let modelConfig = sherpaOnnxOnlineModelConfig(
    tokens: tokens,
    transducer: transducerConfig
  )

  let featConfig = sherpaOnnxFeatureConfig(
    sampleRate: 16000,
    featureDim: 80
  )
  var config = sherpaOnnxKeywordSpotterConfig(
    featConfig: featConfig,
    modelConfig: modelConfig,
    keywordsFile: keywordsFile
  )

  let spotter = SherpaOnnxKeywordSpotterWrapper(config: &config)

  let fileURL: NSURL = NSURL(fileURLWithPath: filePath)
  let audioFile = try! AVAudioFile(forReading: fileURL as URL)

  let audioFormat = audioFile.processingFormat
  assert(audioFormat.sampleRate == 16000)
  assert(audioFormat.channelCount == 1)
  assert(audioFormat.commonFormat == AVAudioCommonFormat.pcmFormatFloat32)

  let audioFrameCount = UInt32(audioFile.length)
  let audioFileBuffer = AVAudioPCMBuffer(pcmFormat: audioFormat, frameCapacity: audioFrameCount)

  try! audioFile.read(into: audioFileBuffer!)
  let array: [Float]! = audioFileBuffer?.array()
  spotter.acceptWaveform(samples: array)

  let tailPadding = [Float](repeating: 0.0, count: 3200)
  spotter.acceptWaveform(samples: tailPadding)

  spotter.inputFinished()
  while spotter.isReady() {
    spotter.decode()
    let keyword = spotter.getResult().keyword
    if keyword != "" {
      print("Detected: \(keyword)")
    }
  }
}

@main
struct App {
  static func main() {
    run()
  }
}
