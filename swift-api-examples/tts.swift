func run() {
  let model = "./vits-piper-en_US-amy-low/en_US-amy-low.onnx"
  let tokens = "./vits-piper-en_US-amy-low/tokens.txt"
  let dataDir = "./vits-piper-en_US-amy-low/espeak-ng-data"
  let vits = sherpaOnnxOfflineTtsVitsModelConfig(
    model: model,
    lexicon: "",
    tokens: tokens,
    dataDir: dataDir
  )
  let modelConfig = sherpaOnnxOfflineTtsModelConfig(vits: vits)
  var ttsConfig = sherpaOnnxOfflineTtsConfig(model: modelConfig)

  let tts = SherpaOnnxOfflineTtsWrapper(config: &ttsConfig)

  let text =
    "“Today as always, men fall into two groups: slaves and free men. Whoever does not have two-thirds of his day for himself, is a slave, whatever he may be: a statesman, a businessman, an official, or a scholar.”"
  let sid = 99
  let speed: Float = 1.0

  let audio = tts.generate(text: text, sid: sid, speed: speed)
  let filename = "test.wav"
  audio.save(filename: filename)

  print("\nSaved to:\n\(filename)")
}

@main
struct App {
  static func main() {
    run()
  }
}
