func run() {
  let model =
    "./sherpa-onnx-w2v-bert-ctc-gl-int8-2026-11-20/model.int8.onnx"
  let tokens =
    "./sherpa-onnx-w2v-bert-ctc-gl-int8-2026-11-20/tokens.txt"

  let w2vBertCtc = sherpaOnnxOfflineW2vBertCtcModelConfig(
    model: model
  )

  let modelConfig = sherpaOnnxOfflineModelConfig(
    tokens: tokens,
    debug: 1,
    w2vBertCtc: w2vBertCtc
  )

  let featConfig = sherpaOnnxFeatureConfig()
  var config = sherpaOnnxOfflineRecognizerConfig(
    featConfig: featConfig,
    modelConfig: modelConfig
  )

  let recognizer = SherpaOnnxOfflineRecognizer(config: &config)

  let filePath = "./sherpa-onnx-w2v-bert-ctc-gl-int8-2026-11-20/test_wavs/0.wav"
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
