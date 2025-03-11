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
  // Please refer to
  // https://github.com/k2-fsa/sherpa-onnx/releases/tag/speech-enhancement-models
  // to download files used in this script
  var config = sherpaOnnxOfflineSpeechDenoiserConfig(
    model: sherpaOnnxOfflineSpeechDenoiserModelConfig(
      gtcrn: sherpaOnnxOfflineSpeechDenoiserGtcrnModelConfig(model: "./gtcrn_simple.onnx"))
  )

  let sd = SherpaOnnxOfflineSpeechDenoiserWrapper(config: &config)

  let fileURL: NSURL = NSURL(fileURLWithPath: "./inp_16k.wav")
  let audioFile = try! AVAudioFile(forReading: fileURL as URL)

  let audioFormat = audioFile.processingFormat
  assert(audioFormat.sampleRate == 16000)
  assert(audioFormat.channelCount == 1)
  assert(audioFormat.commonFormat == AVAudioCommonFormat.pcmFormatFloat32)

  let audioFrameCount = UInt32(audioFile.length)
  let audioFileBuffer = AVAudioPCMBuffer(pcmFormat: audioFormat, frameCapacity: audioFrameCount)

  try! audioFile.read(into: audioFileBuffer!)
  let array: [Float]! = audioFileBuffer?.array()
  let audio = sd.run(samples: array, sampleRate: Int(audioFormat.sampleRate))

  let filename = "enhanced_16k.wav"
  let ok = audio.save(filename: filename)
  if ok == 1 {
    print("\nSaved to:\(filename)")
  } else {
    print("Failed to save to \(filename)")
  }
}

@main
struct App {
  static func main() {
    run()
  }
}
