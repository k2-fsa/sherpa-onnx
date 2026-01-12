func run() {
  let encoderAdaptor =
    "./sherpa-onnx-funasr-nano-int8-2025-12-30/encoder_adaptor.int8.onnx"
  let llm =
    "./sherpa-onnx-funasr-nano-int8-2025-12-30/llm.int8.onnx"
  let embedding =
    "./sherpa-onnx-funasr-nano-int8-2025-12-30/embedding.int8.onnx"
  let tokenizer =
    "./sherpa-onnx-funasr-nano-int8-2025-12-30/Qwen3-0.6B"

  let funasrNano = sherpaOnnxOfflineFunASRNanoModelConfig(
    encoderAdaptor: encoderAdaptor,
    llm: llm,
    embedding: embedding,
    tokenizer: tokenizer
  )

  let modelConfig = sherpaOnnxOfflineModelConfig(
    tokens: "",
    debug: 1,
    funasrNano: funasrNano
  )

  let featConfig = sherpaOnnxFeatureConfig()
  var config = sherpaOnnxOfflineRecognizerConfig(
    featConfig: featConfig,
    modelConfig: modelConfig
  )

  let recognizer = SherpaOnnxOfflineRecognizer(config: &config)

  let filePath = "./sherpa-onnx-funasr-nano-int8-2025-12-30/test_wavs/lyrics.wav"
  let audio = SherpaOnnxWaveWrapper.readWave(filename: filePath)

  let result = recognizer.decode(samples: audio.samples, sampleRate: audio.sampleRate)
  print("decode done")

  print("\nresult is:\n\(result.text)")
  if !result.timestamps.isEmpty {
    print("\ntimestamps is:\n\(result.timestamps)")
  }
}

@main
struct App {
  static func main() {
    run()
  }
}

