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
  var modelConfig: SherpaOnnxOnlineModelConfig
  var modelType = "zipformer2-ctc"
  var filePath: String

  modelType = "transducer"

  if modelType == "transducer" {
    filePath = "./sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20/test_wavs/1.wav"
    let encoder =
      "./sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20/encoder-epoch-99-avg-1.onnx"
    let decoder =
      "./sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20/decoder-epoch-99-avg-1.onnx"
    let joiner =
      "./sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20/joiner-epoch-99-avg-1.onnx"
    let tokens = "./sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20/tokens.txt"

    let transducerConfig = sherpaOnnxOnlineTransducerModelConfig(
      encoder: encoder,
      decoder: decoder,
      joiner: joiner
    )

    modelConfig = sherpaOnnxOnlineModelConfig(
      tokens: tokens,
      transducer: transducerConfig
    )
  } else {
    filePath =
      "./sherpa-onnx-streaming-zipformer-ctc-multi-zh-hans-2023-12-13/test_wavs/DEV_T0000000000.wav"
    let model =
      "./sherpa-onnx-streaming-zipformer-ctc-multi-zh-hans-2023-12-13/ctc-epoch-20-avg-1-chunk-16-left-128.onnx"
    let tokens = "./sherpa-onnx-streaming-zipformer-ctc-multi-zh-hans-2023-12-13/tokens.txt"
    let zipfomer2CtcModelConfig = sherpaOnnxOnlineZipformer2CtcModelConfig(
      model: model
    )

    modelConfig = sherpaOnnxOnlineModelConfig(
      tokens: tokens,
      zipformer2Ctc: zipfomer2CtcModelConfig
    )
  }

  let featConfig = sherpaOnnxFeatureConfig(
    sampleRate: 16000,
    featureDim: 80
  )
  var config = sherpaOnnxOnlineRecognizerConfig(
    featConfig: featConfig,
    modelConfig: modelConfig
  )

  let recognizer = SherpaOnnxRecognizer(config: &config)

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
  recognizer.acceptWaveform(samples: array)

  let tailPadding = [Float](repeating: 0.0, count: 3200)
  recognizer.acceptWaveform(samples: tailPadding)

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
