func run() {
  let model =
    "./sherpa-onnx-fire-red-asr2-ctc-zh_en-int8-2026-02-25/model.int8.onnx"
  let tokens =
    "./sherpa-onnx-fire-red-asr2-ctc-zh_en-int8-2026-02-25/tokens.txt"

  let fireRedAsrCtc = sherpaOnnxOfflineFireRedAsrCtcModelConfig(
    model: model
  )

  let modelConfig = sherpaOnnxOfflineModelConfig(
    tokens: tokens,
    debug: 1,
    fireRedAsrCtc: fireRedAsrCtc
  )

  let featConfig = sherpaOnnxFeatureConfig()
  var config = sherpaOnnxOfflineRecognizerConfig(
    featConfig: featConfig,
    modelConfig: modelConfig
  )

  let recognizer = SherpaOnnxOfflineRecognizer(config: &config)

  let filePath = "./sherpa-onnx-fire-red-asr2-ctc-zh_en-int8-2026-02-25/test_wavs/1.wav"
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
