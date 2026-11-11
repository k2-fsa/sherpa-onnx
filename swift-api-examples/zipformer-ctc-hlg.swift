import AVFoundation

// This file shows how to use a non-streaming Zipformer CTC model with an HLG
// decoding graph via sherpa-onnx's Swift API.
//
// The HLG graph is passed to the C API through ctcFstDecoderConfig.

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
  let dir = "./sherpa-onnx-zipformer-ctc-en-2023-10-02"
  let model = dir + "/model.int8.onnx"
  let tokens = dir + "/tokens.txt"
  let graph = dir + "/HLG.fst"

  let zipformerCtc = sherpaOnnxOfflineZipformerCtcModelConfig(
    model: model
  )

  let modelConfig = sherpaOnnxOfflineModelConfig(
    tokens: tokens,
    debug: 0,
    zipformerCtc: zipformerCtc
  )

  let featConfig = sherpaOnnxFeatureConfig(
    sampleRate: 16000,
    featureDim: 80
  )

  // This is what the example exercises: the graph and maxActive reach the C
  // API through ctcFstDecoderConfig.
  let ctcFstDecoderConfig = sherpaOnnxOfflineCtcFstDecoderConfig(
    graph: graph,
    maxActive: 3000
  )

  var config = sherpaOnnxOfflineRecognizerConfig(
    featConfig: featConfig,
    modelConfig: modelConfig,
    ctcFstDecoderConfig: ctcFstDecoderConfig
  )

  let recognizer = SherpaOnnxOfflineRecognizer(config: &config)

  let filePath = dir + "/test_wavs/0.wav"
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
}

@main
struct App {
  static func main() {
    run()
  }
}
