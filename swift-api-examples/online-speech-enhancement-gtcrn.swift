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
  let model = "./gtcrn_simple.onnx"
  // Please refer to
  // https://github.com/k2-fsa/sherpa-onnx/releases/tag/speech-enhancement-models
  // to download files used in this script
  var config = sherpaOnnxOnlineSpeechDenoiserConfig(
    model: sherpaOnnxOfflineSpeechDenoiserModelConfig(
      gtcrn: sherpaOnnxOfflineSpeechDenoiserGtcrnModelConfig(model: model))
  )

  let sd = SherpaOnnxOnlineSpeechDenoiserWrapper(config: &config)

  let fileURL: NSURL = NSURL(fileURLWithPath: "./inp_16k.wav")
  let audioFile = try! AVAudioFile(forReading: fileURL as URL)

  let audioFormat = audioFile.processingFormat
  assert(audioFormat.sampleRate == 16000)
  assert(audioFormat.channelCount == 1)
  assert(audioFormat.commonFormat == AVAudioCommonFormat.pcmFormatFloat32)

  let audioFrameCount = UInt32(audioFile.length)
  let audioFileBuffer = AVAudioPCMBuffer(pcmFormat: audioFormat, frameCapacity: audioFrameCount)

  try! audioFile.read(into: audioFileBuffer!)
  let samples: [Float]! = audioFileBuffer?.array()

  var enhanced: [Float] = []
  let frameShift = sd.frameShiftInSamples

  var start = 0
  while start < samples.count {
    let end = min(start + frameShift, samples.count)
    let audio = sd.run(samples: Array(samples[start..<end]), sampleRate: Int(audioFormat.sampleRate))
    enhanced.append(contentsOf: audio.samples)
    start = end
  }

  enhanced.append(contentsOf: sd.flush().samples)

  let filename = "enhanced-online-gtcrn.wav"
  _ = enhanced.withUnsafeBufferPointer { p in
    SherpaOnnxWriteWave(
      p.baseAddress,
      Int32(enhanced.count),
      Int32(sd.sampleRate),
      toCPointer(filename))
  }
  print("\nSaved to:\(filename)")
}

@main
struct App {
  static func main() {
    run()
  }
}
