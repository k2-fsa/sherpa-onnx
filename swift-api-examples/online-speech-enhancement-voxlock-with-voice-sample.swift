// Copyright (c)  2026  Xiaomi Corporation
//
// This file shows how to use the online speech enhancement API with VoxLock
// models and a voice sample.
//
// VoxLock is a target-speaker voice isolation model. It keeps the voice of one
// speaker, e.g., the caller in a phone call, and removes the other voices.
//
// The model is from https://github.com/arbazsiddiqui/voxlock
//
// This example provides a voice sample of the speaker to keep, so VoxLock
// isolates that speaker from the first sample. The voice sample must have at
// least 3 seconds of the speaker's voice; from a longer file, VoxLock uses the
// loudest 3 seconds within its first 30 seconds. See
// ./online-speech-enhancement-voxlock.swift if you don't have a voice sample.
//
// Please refer to
// https://github.com/k2-fsa/sherpa-onnx/releases/tag/speech-enhancement-models
// to download files used in this script
//
// wget https://github.com/k2-fsa/sherpa-onnx/releases/download/speech-enhancement-models/sherpa-onnx-voxlock-int8.tar.bz2
// tar xvf sherpa-onnx-voxlock-int8.tar.bz2

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
  var config = sherpaOnnxOnlineSpeechDenoiserConfig(
    model: sherpaOnnxOfflineSpeechDenoiserModelConfig(
      voxlock: sherpaOnnxOfflineSpeechDenoiserVoxLockModelConfig(
        enrollModel: "./sherpa-onnx-voxlock-int8/enroll.onnx",
        chunkModel: "./sherpa-onnx-voxlock-int8/chunk_int8.onnx",
        voiceSample: "./sherpa-onnx-voxlock-int8/test_wavs/en-voice-sample-f.wav")))

  let sd = SherpaOnnxOnlineSpeechDenoiserWrapper(config: &config)

  let fileURL: NSURL =
    NSURL(fileURLWithPath: "./sherpa-onnx-voxlock-int8/test_wavs/en-combined.wav")
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

  let begin = Date()
  var start = 0
  while start < samples.count {
    let end = min(start + frameShift, samples.count)
    let audio = sd.run(samples: Array(samples[start..<end]), sampleRate: Int(audioFormat.sampleRate))
    enhanced.append(contentsOf: audio.samples)
    start = end
  }

  enhanced.append(contentsOf: sd.flush().samples)
  let elapsedSeconds = Date().timeIntervalSince(begin)

  let filename = "voxlock-enhanced-online-with-voice-sample-16k.wav"
  _ = enhanced.withUnsafeBufferPointer { p in
    SherpaOnnxWriteWave(
      p.baseAddress,
      Int32(enhanced.count),
      Int32(sd.sampleRate),
      toCPointer(filename))
  }
  print("\nSaved to:\(filename)")

  let duration = Double(samples.count) / audioFormat.sampleRate
  print(String(format: "Elapsed seconds: %.3f", elapsedSeconds))
  print(String(format: "Audio duration in seconds: %.3f", duration))
  print(String(format: "RTF: %.3f/%.3f = %.3f", elapsedSeconds, duration, elapsedSeconds / duration))
}

@main
struct App {
  static func main() {
    run()
  }
}
