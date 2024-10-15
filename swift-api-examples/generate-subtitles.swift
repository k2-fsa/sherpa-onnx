/*
This file shows how to use Swift API to generate subtitles.

You can use the files from
https://huggingface.co/csukuangfj/vad/tree/main
for testing.

For instance, to generate subtitles for Obama.mov, please first
use

ffmpeg -i ./Obama.mov -acodec pcm_s16le -ac 1 -ar 16000 Obama.wav

to extract the audio part from the video.

This file supports only processing WAV sound files, so you have to first
extract audios from videos.

Please see
./run-generate-subtitles.sh
for usages.
*/

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

extension TimeInterval {
  var hourMinuteSecondMS: String {
    String(format: "%d:%02d:%02d,%03d", hour, minute, second, millisecond)
  }

  var hour: Int {
    Int((self / 3600).truncatingRemainder(dividingBy: 3600))
  }
  var minute: Int {
    Int((self / 60).truncatingRemainder(dividingBy: 60))
  }
  var second: Int {
    Int(truncatingRemainder(dividingBy: 60))
  }
  var millisecond: Int {
    Int((self * 1000).truncatingRemainder(dividingBy: 1000))
  }
}

extension String {
  var fileURL: URL {
    return URL(fileURLWithPath: self)
  }
  var pathExtension: String {
    return fileURL.pathExtension
  }
  var lastPathComponent: String {
    return fileURL.lastPathComponent
  }
  var stringByDeletingPathExtension: String {
    return fileURL.deletingPathExtension().path
  }
}

class SpeechSegment: CustomStringConvertible {

  let start: Float
  let end: Float
  let text: String

  init(start: Float, duration: Float, text: String) {
    self.start = start
    self.end = start + duration
    self.text = text
  }
  public var description: String {
    var s: String
    s = TimeInterval(self.start).hourMinuteSecondMS
    s += " --> "
    s += TimeInterval(self.end).hourMinuteSecondMS
    s += "\n"
    s += self.text

    return s
  }
}

func run() {
  var recognizer: SherpaOnnxOfflineRecognizer
  var modelConfig: SherpaOnnxOfflineModelConfig
  var modelType = "whisper"
  // modelType = "paraformer"
  var filePath = "/Users/fangjun/Desktop/Obama.wav"  // English
  // filePath = "/Users/fangjun/Desktop/lei-jun.wav"  // Chinese
  // please go to https://huggingface.co/csukuangfj/vad
  // to download the above two files

  if modelType == "whisper" {
    // for English
    let encoder = "./sherpa-onnx-whisper-tiny.en/tiny.en-encoder.int8.onnx"
    let decoder = "./sherpa-onnx-whisper-tiny.en/tiny.en-decoder.int8.onnx"
    let tokens = "./sherpa-onnx-whisper-tiny.en/tiny.en-tokens.txt"

    let whisperConfig = sherpaOnnxOfflineWhisperModelConfig(
      encoder: encoder,
      decoder: decoder
    )

    modelConfig = sherpaOnnxOfflineModelConfig(
      tokens: tokens,
      whisper: whisperConfig,
      debug: 0,
      modelType: "whisper"
    )
  } else if modelType == "paraformer" {
    // for Chinese
    let model = "./sherpa-onnx-paraformer-zh-2023-09-14/model.int8.onnx"
    let tokens = "./sherpa-onnx-paraformer-zh-2023-09-14/tokens.txt"
    let paraformerConfig = sherpaOnnxOfflineParaformerModelConfig(
      model: model
    )

    modelConfig = sherpaOnnxOfflineModelConfig(
      tokens: tokens,
      paraformer: paraformerConfig,
      debug: 0,
      modelType: "paraformer"
    )
  } else {
    print("Please specify a supported modelType \(modelType)")
    return
  }

  let sampleRate = 16000
  let featConfig = sherpaOnnxFeatureConfig(
    sampleRate: sampleRate,
    featureDim: 80
  )
  var config = sherpaOnnxOfflineRecognizerConfig(
    featConfig: featConfig,
    modelConfig: modelConfig
  )

  recognizer = SherpaOnnxOfflineRecognizer(config: &config)

  let audioFile = try! AVAudioFile(forReading: filePath.fileURL)

  let audioFormat = audioFile.processingFormat
  assert(audioFormat.sampleRate == Double(sampleRate))
  assert(audioFormat.channelCount == 1)
  assert(audioFormat.commonFormat == AVAudioCommonFormat.pcmFormatFloat32)

  let sileroVadConfig = sherpaOnnxSileroVadModelConfig(
    model: "./silero_vad.onnx"
  )

  var vadModelConfig = sherpaOnnxVadModelConfig(sileroVad: sileroVadConfig)
  let vad = SherpaOnnxVoiceActivityDetectorWrapper(
    config: &vadModelConfig, buffer_size_in_seconds: 120)

  let audioFrameCount = UInt32(audioFile.length)
  let audioFileBuffer = AVAudioPCMBuffer(pcmFormat: audioFormat, frameCapacity: audioFrameCount)

  try! audioFile.read(into: audioFileBuffer!)
  var array: [Float]! = audioFileBuffer?.array()

  let windowSize = Int(vadModelConfig.silero_vad.window_size)

  var segments: [SpeechSegment] = []

  for offset in stride(from: 0, to: array.count, by: windowSize) {
    let end = min(offset + windowSize, array.count)
    vad.acceptWaveform(samples: [Float](array[offset..<end]))
  }

  vad.flush()
  var index: Int = 0
  while !vad.isEmpty() {
    let s = vad.front()
    vad.pop()
    let result = recognizer.decode(samples: s.samples)

    segments.append(
      SpeechSegment(
        start: Float(s.start) / Float(sampleRate),
        duration: Float(s.samples.count) / Float(sampleRate),
        text: result.text))

    print(segments.last!)
  }

  let srt: String = zip(segments.indices, segments).map { (index, element) in
    return "\(index+1)\n\(element)"
  }.joined(separator: "\n\n")

  let srtFilename: String = filePath.stringByDeletingPathExtension + ".srt"
  do {
    try srt.write(to: srtFilename.fileURL, atomically: true, encoding: .utf8)
  } catch {
    print("Error writing: \(error.localizedDescription)")
  }

  print("Saved to \(srtFilename)")
}

@main
struct App {
  static func main() {
    run()
  }
}
