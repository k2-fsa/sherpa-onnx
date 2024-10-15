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
  let segmentationModel = "./sherpa-onnx-pyannote-segmentation-3-0/model.onnx"
  let embeddingExtractorModel = "./3dspeaker_speech_eres2net_base_sv_zh-cn_3dspeaker_16k.onnx"
  let waveFilename = "./0-four-speakers-zh.wav"

  // There are 4 speakers in ./0-four-speakers-zh.wav, so we use 4 here
  let numSpeakers = 4
  var config = sherpaOnnxOfflineSpeakerDiarizationConfig(
    segmentation: sherpaOnnxOfflineSpeakerSegmentationModelConfig(
      pyannote: sherpaOnnxOfflineSpeakerSegmentationPyannoteModelConfig(model: segmentationModel)),
    embedding: sherpaOnnxSpeakerEmbeddingExtractorConfig(model: embeddingExtractorModel),
    clustering: sherpaOnnxFastClusteringConfig(numClusters: numSpeakers)
  )

  let sd = SherpaOnnxOfflineSpeakerDiarizationWrapper(config: &config)

  let fileURL: NSURL = NSURL(fileURLWithPath: waveFilename)
  let audioFile = try! AVAudioFile(forReading: fileURL as URL)

  let audioFormat = audioFile.processingFormat
  assert(Int(audioFormat.sampleRate) == sd.sampleRate)
  assert(audioFormat.channelCount == 1)
  assert(audioFormat.commonFormat == AVAudioCommonFormat.pcmFormatFloat32)

  let audioFrameCount = UInt32(audioFile.length)
  let audioFileBuffer = AVAudioPCMBuffer(pcmFormat: audioFormat, frameCapacity: audioFrameCount)

  try! audioFile.read(into: audioFileBuffer!)
  let array: [Float]! = audioFileBuffer?.array()
  print("Started!")
  let segments = sd.process(samples: array)
  for i in 0..<segments.count {
    print("\(segments[i].start) -- \(segments[i].end) speaker_\(segments[i].speaker)")
  }
}

@main
struct App {
  static func main() {
    run()
  }
}
