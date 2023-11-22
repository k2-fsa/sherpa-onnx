func run() {
  let model = "./vits-vctk/vits-vctk.onnx"
  let lexicon = "./vits-vctk/lexicon.txt"
  let tokens = "./vits-vctk/tokens.txt"
  let vits = sherpaOnnxOfflineTtsVitsModelConfig(
    model: model,
    lexicon: lexicon,
    tokens: tokens
  )
  let modelConfig = sherpaOnnxOfflineTtsModelConfig(vits: vits)
  var ttsConfig = sherpaOnnxOfflineTtsConfig(model: modelConfig)

  let tts = SherpaOnnxOfflineTtsWrapper(config: &ttsConfig)

  let text = "How are you doing? Fantastic!"
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
