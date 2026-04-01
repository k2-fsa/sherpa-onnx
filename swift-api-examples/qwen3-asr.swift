func run() {
  let convFrontend =
    "./sherpa-onnx-qwen3-asr-0.6B-int8-2026-03-25/conv_frontend.onnx"
  let encoder =
    "./sherpa-onnx-qwen3-asr-0.6B-int8-2026-03-25/encoder.int8.onnx"
  let decoder =
    "./sherpa-onnx-qwen3-asr-0.6B-int8-2026-03-25/decoder.int8.onnx"
  let tokenizer =
    "./sherpa-onnx-qwen3-asr-0.6B-int8-2026-03-25/tokenizer"

  let qwen3Asr = sherpaOnnxOfflineQwen3ASRModelConfig(
    convFrontend: convFrontend,
    encoder: encoder,
    decoder: decoder,
    tokenizer: tokenizer,
    maxTotalLen: 512,
    maxNewTokens: 128,
    temperature: 1e-6,
    topP: 0.8,
    seed: 42,
    hotwords: ""
  )

  let modelConfig = sherpaOnnxOfflineModelConfig(
    tokens: "",
    numThreads: 2,
    provider: "cpu",
    debug: 0,
    qwen3Asr: qwen3Asr
  )

  let featConfig = sherpaOnnxFeatureConfig()
  var config = sherpaOnnxOfflineRecognizerConfig(
    featConfig: featConfig,
    modelConfig: modelConfig
  )

  let recognizer = SherpaOnnxOfflineRecognizer(config: &config)

  let filePath =
    "./sherpa-onnx-qwen3-asr-0.6B-int8-2026-03-25/test_wavs/raokouling.wav"
  let audio = SherpaOnnxWaveWrapper.readWave(filename: filePath)

  let result = recognizer.decode(samples: audio.samples, sampleRate: audio.sampleRate)
  print("decode done")

  print("\nresult is:\n\(result.text)")
}

@main
struct App {
  static func main() {
    run()
  }
}
