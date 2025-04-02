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
  let model = "./sherpa-onnx-dolphin-base-ctc-multi-lang-int8-2025-04-02/model.int8.onnx"
  let tokens = "./sherpa-onnx-dolphin-base-ctc-multi-lang-int8-2025-04-02/tokens.txt"

  let dolphin = sherpaOnnxOfflineDolphinModelConfig(
    model: model
  )

  let modelConfig = sherpaOnnxOfflineModelConfig(
    tokens: tokens,
    debug: 0,
    dolphin: dolphin
  )

  let featConfig = sherpaOnnxFeatureConfig(
    sampleRate: 16000,
    featureDim: 80
  )
  var config = sherpaOnnxOfflineRecognizerConfig(
    featConfig: featConfig,
    modelConfig: modelConfig
  )

  let recognizer = SherpaOnnxOfflineRecognizer(config: &config)

  let filePath = "./sherpa-onnx-dolphin-base-ctc-multi-lang-int8-2025-04-02/test_wavs/0.wav"
  let fileURL: NSURL = NSURL(fileURLWithPath: filePath)
  let audioFile = try! AVAudioFile(forReading: fileURL as URL)

  let audioFormat = audioFile.processingFormat
  assert(audioFormat.channelCount == 1)
  assert(audioFormat.commonFormat == AVAudioCommonFormat.pcmFormatFloat32)

  let audioFrameCount = UInt32(audioFile.length)
  let audioFileBuffer = AVAudioPCMBuffer(pcmFormat: audioFormat, frameCapacity: audioFrameCount)

  try! audioFile.read(into: audioFileBuffer!)
  let array: [Float]! = audioFileBuffer?.array()
  let result = recognizer.decode(samples: array, sampleRate: Int(audioFormat.sampleRate))
  print("\nresult is:\n\(result.text)")
  if result.timestamps.count != 0 {
    print("\ntimestamps is:\n\(result.timestamps)")
  }

}

@main
struct App {
  static func main() {
    run()
  }
}
