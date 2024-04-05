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
  let filePath =
    "./sherpa-onnx-streaming-zipformer-ctc-small-2024-03-18/test_wavs/8k.wav"
  let model =
    "./sherpa-onnx-streaming-zipformer-ctc-small-2024-03-18/ctc-epoch-30-avg-3-chunk-16-left-128.int8.onnx"
  let tokens = "./sherpa-onnx-streaming-zipformer-ctc-small-2024-03-18/tokens.txt"
  let zipfomer2CtcModelConfig = sherpaOnnxOnlineZipformer2CtcModelConfig(
    model: model
  )

  let modelConfig = sherpaOnnxOnlineModelConfig(
    tokens: tokens,
    zipformer2Ctc: zipfomer2CtcModelConfig
  )

  let featConfig = sherpaOnnxFeatureConfig(
    sampleRate: 16000,
    featureDim: 80
  )

  let ctcFstDecoderConfig = sherpaOnnxOnlineCtcFstDecoderConfig(
    graph: "./sherpa-onnx-streaming-zipformer-ctc-small-2024-03-18/HLG.fst",
    maxActive: 3000
  )

  var config = sherpaOnnxOnlineRecognizerConfig(
    featConfig: featConfig,
    modelConfig: modelConfig,
    ctcFstDecoderConfig: ctcFstDecoderConfig
  )

  let recognizer = SherpaOnnxRecognizer(config: &config)

  let fileURL: NSURL = NSURL(fileURLWithPath: filePath)
  let audioFile = try! AVAudioFile(forReading: fileURL as URL)

  let audioFormat = audioFile.processingFormat
  assert(audioFormat.channelCount == 1)
  assert(audioFormat.commonFormat == AVAudioCommonFormat.pcmFormatFloat32)

  let audioFrameCount = UInt32(audioFile.length)
  let audioFileBuffer = AVAudioPCMBuffer(pcmFormat: audioFormat, frameCapacity: audioFrameCount)

  try! audioFile.read(into: audioFileBuffer!)
  let array: [Float]! = audioFileBuffer?.array()
  recognizer.acceptWaveform(samples: array, sampleRate: Int(audioFormat.sampleRate))

  let tailPadding = [Float](repeating: 0.0, count: 3200)
  recognizer.acceptWaveform(samples: tailPadding, sampleRate: Int(audioFormat.sampleRate))

  recognizer.inputFinished()
  while recognizer.isReady() {
    recognizer.decode()
  }

  let result = recognizer.getResult()
  print("\nresult is:\n\(result.text)")
}

@main
struct App {
  static func main() {
    run()
  }
}
