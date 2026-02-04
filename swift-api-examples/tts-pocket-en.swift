/// PocketTTS Swift demo using SherpaOnnx C API
/// Both modes: with progress callback and direct generation

// -------------------------
// 1️⃣ Callback handler
// -------------------------
class PocketTtsProgressHandler {
  func progress(samples: [Float], progress: Float) {
    print(String(format: "Progress: %.2f%%", progress * 100))
  }
}

// -------------------------
// 2️⃣ Main demo function
// -------------------------
func runPocketTtsDemo() {
  // -------------------------
  // Load Pocket model
  // -------------------------
  let pocket = sherpaOnnxOfflineTtsPocketModelConfig(
    lmFlow: "./sherpa-onnx-pocket-tts-int8-2026-01-26/lm_flow.int8.onnx",
    lmMain: "./sherpa-onnx-pocket-tts-int8-2026-01-26/lm_main.int8.onnx",
    encoder: "./sherpa-onnx-pocket-tts-int8-2026-01-26/encoder.onnx",
    decoder: "./sherpa-onnx-pocket-tts-int8-2026-01-26/decoder.int8.onnx",
    textConditioner: "./sherpa-onnx-pocket-tts-int8-2026-01-26/text_conditioner.onnx",
    vocabJson: "./sherpa-onnx-pocket-tts-int8-2026-01-26/vocab.json",
    tokenScoresJson: "./sherpa-onnx-pocket-tts-int8-2026-01-26/token_scores.json"
  )

  let modelConfig = sherpaOnnxOfflineTtsModelConfig(numThreads: 2, pocket: pocket)
  var ttsConfig = sherpaOnnxOfflineTtsConfig(model: modelConfig)
  ttsConfig.model.debug = 0

  // -------------------------
  // Create TTS wrapper
  // -------------------------
  let tts = SherpaOnnxOfflineTtsWrapper(config: &ttsConfig)

  // -------------------------
  // Load reference audio
  // -------------------------
  let referenceAudioFile = "./sherpa-onnx-pocket-tts-int8-2026-01-26/test_wavs/bria.wav"
  let referenceWave = SherpaOnnxWaveWrapper.readWave(filename: referenceAudioFile)

  // -------------------------
  // Build Swift generation config
  // -------------------------
  var genConfig = SherpaOnnxGenerationConfigSwift()
  genConfig.speed = 1.0
  genConfig.referenceAudio = referenceWave.samples
  genConfig.referenceSampleRate = referenceWave.sampleRate
  genConfig.extra = ["max_reference_audio_len": "10.0"]

  // -------------------------
  // Input text
  // -------------------------
  let text = """
    Today as always, men fall into two groups: slaves and free men. Whoever \
    does not have two-thirds of his day for himself, is a slave, whatever \
    he may be: a statesman, a businessman, an official, or a scholar. \
    Friends fell out often because life was changing so fast. \
    The easiest thing in the world was to lose touch with someone.
    """

  // -------------------------
  // Use if(true) to switch modes
  // -------------------------
  if true {
    // -------------------------
    // 1️⃣ Mode A: Progress callback
    // -------------------------
    let progressHandler = PocketTtsProgressHandler()
    let arg = Unmanaged.passUnretained(progressHandler).toOpaque()

    let callback: TtsProgressCallbackWithArg = { samples, n, progress, arg in
      let handler = Unmanaged<PocketTtsProgressHandler>.fromOpaque(arg!).takeUnretainedValue()
      var buffer: [Float] = []
      if let samples {
        for i in 0..<n {
          buffer.append(samples[Int(i)])
        }
      }
      handler.progress(samples: buffer, progress: progress)
      return 1  // continue generating
    }

    let audio = tts.generateWithConfig(
      text: text,
      config: genConfig,
      callback: callback,
      arg: arg
    )

    let outputFile = "generated-pocket-callback.wav"
    if audio.save(filename: outputFile) == 1 {
      print("Saved to: \(outputFile)")
    } else {
      print("Failed to save to \(outputFile)")
    }

  } else {
    // -------------------------
    // 2️⃣ Mode B: Direct generation (no callback)
    // -------------------------
    let audio = tts.generateWithConfig(
      text: text,
      config: genConfig,
      callback: nil,
      arg: nil
    )

    let outputFile = "generated-pocket-direct.wav"
    if audio.save(filename: outputFile) == 1 {
      print("Saved to: \(outputFile)")
    } else {
      print("Failed to save to \(outputFile)")
    }
  }
}

// -------------------------
// Run demo
// -------------------------
@main
struct App {
  static func main() {
    runPocketTtsDemo()
  }
}
