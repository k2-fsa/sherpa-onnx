import Foundation

class SupertonicTtsProgressHandler {
  func progress(samples: [Float], progress: Float) {
    print(String(format: "Received %d samples, Progress: %.2f%%", samples.count, progress * 100))
  }
}

func runSupertonicTtsDemo() {
  let supertonic = sherpaOnnxOfflineTtsSupertonicModelConfig(
    durationPredictor: "./sherpa-onnx-supertonic-tts-int8-2026-03-06/duration_predictor.int8.onnx",
    textEncoder: "./sherpa-onnx-supertonic-tts-int8-2026-03-06/text_encoder.int8.onnx",
    vectorEstimator: "./sherpa-onnx-supertonic-tts-int8-2026-03-06/vector_estimator.int8.onnx",
    vocoder: "./sherpa-onnx-supertonic-tts-int8-2026-03-06/vocoder.int8.onnx",
    ttsJson: "./sherpa-onnx-supertonic-tts-int8-2026-03-06/tts.json",
    unicodeIndexer: "./sherpa-onnx-supertonic-tts-int8-2026-03-06/unicode_indexer.bin",
    voiceStyle: "./sherpa-onnx-supertonic-tts-int8-2026-03-06/voice.bin"
  )

  let modelConfig = sherpaOnnxOfflineTtsModelConfig(numThreads: 2, supertonic: supertonic)
  var ttsConfig = sherpaOnnxOfflineTtsConfig(model: modelConfig)
  ttsConfig.model.debug = 1

  let tts = SherpaOnnxOfflineTtsWrapper(config: &ttsConfig)

  var genConfig = SherpaOnnxGenerationConfigSwift()
  genConfig.sid = 6
  genConfig.numSteps = 5
  genConfig.speed = 1.25
  genConfig.extra = ["lang": "en"]

  let text =
    "Today as always, men fall into two groups: slaves and free men. Whoever "
    + "does not have two-thirds of his day for himself, is a slave, whatever "
    + "he may be: a statesman, a businessman, an official, or a scholar."

  func generateAndSave(
    outputFile: String, callback: TtsProgressCallbackWithArg? = nil,
    arg: UnsafeMutableRawPointer? = nil
  ) {
    let audio = tts.generateWithConfig(
      text: text,
      config: genConfig,
      callback: callback,
      arg: arg
    )

    if audio.save(filename: outputFile) == 1 {
      print("Saved to: \(outputFile)")
    } else {
      print("Failed to save to \(outputFile)")
    }
  }

  // -------------------------
  // Option 1: with callback
  // -------------------------
  let useCallback = true
  if useCallback {
    let progressHandler = SupertonicTtsProgressHandler()
    let arg = Unmanaged.passUnretained(progressHandler).toOpaque()

    let callback: TtsProgressCallbackWithArg = { samples, n, progress, arg in
      let handler = Unmanaged<SupertonicTtsProgressHandler>.fromOpaque(arg!).takeUnretainedValue()

      let buffer: [Float] =
        samples != nil ? Array(UnsafeBufferPointer(start: samples, count: Int(n))) : []
      handler.progress(samples: buffer, progress: progress)
      return 1  // continue generating
    }

    generateAndSave(outputFile: "generated-supertonic-callback.wav", callback: callback, arg: arg)
  } else {
    // -------------------------
    // Option 2: direct generation
    // -------------------------
    generateAndSave(outputFile: "generated-supertonic-direct.wav")
  }
}

// -------------------------
// Run demo
// -------------------------
@main
struct App {
  static func main() {
    runSupertonicTtsDemo()
  }
}
