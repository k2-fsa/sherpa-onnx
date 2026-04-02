func run() {
  let encoder =
    "./sherpa-onnx-cohere-transcribe-14-lang-int8-2026-04-01/encoder.int8.onnx"
  let decoder =
    "./sherpa-onnx-cohere-transcribe-14-lang-int8-2026-04-01/decoder.int8.onnx"
  let tokens =
    "./sherpa-onnx-cohere-transcribe-14-lang-int8-2026-04-01/tokens.txt"

  let cohereTranscribe = sherpaOnnxOfflineCohereTranscribeModelConfig(
    encoder: encoder,
    decoder: decoder,
    usePunct: true,
    useInverseTextNormalization: true
  )

  let modelConfig = sherpaOnnxOfflineModelConfig(
    tokens: tokens,
    numThreads: 1,
    provider: "cpu",
    debug: 0,
    cohereTranscribe: cohereTranscribe
  )

  let featConfig = sherpaOnnxFeatureConfig()
  var config = sherpaOnnxOfflineRecognizerConfig(
    featConfig: featConfig,
    modelConfig: modelConfig
  )

  let recognizer = SherpaOnnxOfflineRecognizer(config: &config)

  let filePath =
    "./sherpa-onnx-cohere-transcribe-14-lang-int8-2026-04-01/test_wavs/en.wav"
  let audio = SherpaOnnxWaveWrapper.readWave(filename: filePath)

  let stream = recognizer.createStream()
  stream.setOption(key: "language", value: "en")
  stream.acceptWaveform(samples: audio.samples, sampleRate: audio.sampleRate)

  recognizer.decode(stream: stream)

  let result = recognizer.getResult(stream: stream)
  print("decode done")

  print("\nresult is:\n\(result.text)")
}

@main
struct App {
  static func main() {
    run()
  }
}
