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
  let encoder = "./sherpa-onnx-whisper-tiny/tiny-encoder.int8.onnx"
  let decoder = "./sherpa-onnx-whisper-tiny/tiny-decoder.int8.onnx"

  let whisperConfig = sherpaOnnxSpokenLanguageIdentificationWhisperConfig(
    encoder: encoder,
    decoder: decoder
  )

  var config = sherpaOnnxSpokenLanguageIdentificationConfig(
    whisper: whisperConfig,
    numThreads: 1,
    debug: 1,
    provider: "cpu"
  )
  let filePath = "./sherpa-onnx-whisper-tiny/test_wavs/0.wav"

  let slid = SherpaOnnxSpokenLanguageIdentificationWrapper(config: &config)

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
  let result = slid.decode(samples: array)

  print("\nDetectedllanguage is:\n\(result.lang)")
}

@main
struct App {
  static func main() {
    run()
  }
}
