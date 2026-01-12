func run() {
  let funasrNano = sherpaOnnxOfflineFunASRNanoModelConfig(
    encoderAdaptor: "./sherpa-onnx-funasr-nano-int8-2025-12-30/encoder_adaptor.int8.onnx",
    llm: "./sherpa-onnx-funasr-nano-int8-2025-12-30/llm.int8.onnx",
    embedding: "./sherpa-onnx-funasr-nano-int8-2025-12-30/embedding.int8.onnx",
    tokenizer: "./sherpa-onnx-funasr-nano-int8-2025-12-30/Qwen3-0.6B",
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

  let filePath = "./sherpa-onnx-funasr-nano-int8-2025-12-30/test_wavs/dia_yue.wav"
  let audio = SherpaOnnxWaveWrapper.readWave(filename: filePath)

  let result = recognizer.decode(samples: audio.samples, sampleRate: audio.sampleRate)
  print("decode done")

  print("\nresult is:\n\(result.text)")
  if result.timestamps.count != 0 {
    print("\ntimestamps is:\n\(result.timestamps)")
  }
}

@main
struct App {
  static func main() {
    run()
  }
}
