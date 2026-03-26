// swift-api-examples/source-separation-uvr.swift
//
// Copyright (c)  2026  Xiaomi Corporation

// This file demonstrates how to use source separation with the
// UVR (MDX-Net) model.
//
// Please refer to
// https://github.com/k2-fsa/sherpa-onnx/releases/tag/source-separation-models
// to download files used in this script

func run() {
  var config = sherpaOnnxOfflineSourceSeparationConfig(
    model: sherpaOnnxOfflineSourceSeparationModelConfig(
      uvr: sherpaOnnxOfflineSourceSeparationUvrModelConfig(
        model: "./UVR-MDX-NET-Voc_FT.onnx"
      ),
      numThreads: 1,
      debug: 1
    )
  )

  let ss = SherpaOnnxOfflineSourceSeparationWrapper(config: &config)

  let wave = SherpaOnnxOfflineSourceSeparationWrapper.readWave(
    filename: "./qi-feng-le-zh.wav")
  print(
    "Input: channels=\(wave.numChannels), samples=\(wave.numSamples), sampleRate=\(wave.sampleRate)"
  )

  let output = ss.process(wave: wave)

  print("Output: \(output.numStems) stems, sampleRate=\(output.sampleRate)")

  let stemNames = ["uvr-vocals", "uvr-non-vocals"]
  for i in 0..<min(Int(output.numStems), stemNames.count) {
    let filename = "\(stemNames[i]).wav"
    let ok = output.saveStem(stemIndex: i, filename: filename)
    if ok == 1 {
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
