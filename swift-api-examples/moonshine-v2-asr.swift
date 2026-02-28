func run() {
  let encoder =
    "./sherpa-onnx-moonshine-tiny-en-quantized-2026-02-27/encoder_model.ort"
  let decoder =
    "./sherpa-onnx-moonshine-tiny-en-quantized-2026-02-27/decoder_model_merged.ort"
  let tokens =
    "./sherpa-onnx-moonshine-tiny-en-quantized-2026-02-27/tokens.txt"

  let moonshine = sherpaOnnxOfflineMoonshineModelConfig(
    encoder: encoder,
    mergedDecoder: decoder
  )

  let modelConfig = sherpaOnnxOfflineModelConfig(
    tokens: tokens,
    debug: 1,
    moonshine: moonshine
  )

  let featConfig = sherpaOnnxFeatureConfig()
  var config = sherpaOnnxOfflineRecognizerConfig(
    featConfig: featConfig,
    modelConfig: modelConfig
  )

  let recognizer = SherpaOnnxOfflineRecognizer(config: &config)

  let filePath = "./sherpa-onnx-moonshine-tiny-en-quantized-2026-02-27/test_wavs/0.wav"
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
