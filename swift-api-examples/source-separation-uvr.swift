// swift-api-examples/source-separation-uvr.swift
//
// Copyright (c)  2026  Xiaomi Corporation

// This file demonstrates how to use source separation with the
// UVR (MDX-Net) model.
//
// Please refer to
// https://github.com/k2-fsa/sherpa-onnx/releases/tag/source-separation-models
// to download files used in this script

import Foundation

func run() {
  let config = SourceSeparationConfig(
    uvr: .init(model: "./UVR-MDX-NET-Voc_FT.onnx"),
    numThreads: 1,
    debug: true
  )

  guard let separator = SourceSeparator(config: config) else {
    print("Failed to create SourceSeparator")
    return
  }

  guard let input = AudioData(filename: "./qi-feng-le-zh.wav") else {
    print("Failed to read ./qi-feng-le-zh.wav")
    return
  }
  print(
    "Input: channels=\(input.channelCount), samples=\(input.samplesPerChannel), sampleRate=\(input.sampleRate)"
  )

  let start = CFAbsoluteTimeGetCurrent()
  guard let stems = separator.process(buffer: input) else {
    print("Source separation failed")
    return
  }
  let elapsed = CFAbsoluteTimeGetCurrent() - start
  let audioDuration = Double(input.samplesPerChannel) / Double(input.sampleRate)
  let rtf = elapsed / audioDuration

  print("Output: \(stems.count) stems, sampleRate=\(input.sampleRate)")
  print(
    "Elapsed: \(String(format: "%.2f", elapsed))s, Audio: \(String(format: "%.2f", audioDuration))s, RTF = \(String(format: "%.3f", rtf))"
  )

  let stemNames = ["uvr-vocals", "uvr-non-vocals"]
  for i in 0..<min(stems.count, stemNames.count) {
    let filename = "\(stemNames[i]).wav"
    if stems[i].save(to: filename) {
      print("Saved \(filename)")
    } else {
      print("Failed to save \(filename)")
    }
  }
}

@main
struct App {
  static func main() {
    run()
  }
}
