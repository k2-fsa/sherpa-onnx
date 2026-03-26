// swift-api-examples/source-separation-spleeter.swift
//
// Copyright (c)  2026  Xiaomi Corporation

// This file demonstrates how to use source separation with the
// Spleeter 2-stems model (UNet-based).
//
// Please refer to
// https://github.com/k2-fsa/sherpa-onnx/releases/tag/source-separation-models
// to download files used in this script

func run() {
  let modelDir = "./sherpa-onnx-spleeter-2stems-fp16"

  var config = sherpaOnnxOfflineSourceSeparationConfig(
    model: sherpaOnnxOfflineSourceSeparationModelConfig(
      spleeter: sherpaOnnxOfflineSourceSeparationSpleeterModelConfig(
        vocals: "\(modelDir)/vocals.fp16.onnx",
        accompaniment: "\(modelDir)/accompaniment.fp16.onnx"
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

  let stemNames = ["vocals", "accompaniment"]
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
