class MyClass {
  func playSamples(samples: [Float]) {
    print("Play \(samples.count) samples")
  }
}

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

  let myClass = MyClass()

  // We use Unretained here so myClass must be kept alive as the callback is invoked
  //
  // See also
  // https://medium.com/codex/swift-c-callback-interoperability-6d57da6c8ee6
  let arg = Unmanaged<MyClass>.passUnretained(myClass).toOpaque()

  let callback: TtsProgressCallbackWithArg = { samples, n, progress, arg in
    let o = Unmanaged<MyClass>.fromOpaque(arg!).takeUnretainedValue()
    var savedSamples: [Float] = []
    for index in 0..<n {
      savedSamples.append(samples![Int(index)])
    }

    o.playSamples(samples: savedSamples)

    // return 1 so that it continues generating
    return 1
  }

  let tts = SherpaOnnxOfflineTtsWrapper(config: &ttsConfig)

  let text =
    "“Today as always, men fall into two groups: slaves and free men. Whoever does not have two-thirds of his day for himself, is a slave, whatever he may be: a statesman, a businessman, an official, or a scholar.”"
  var genConfig = SherpaOnnxGenerationConfigSwift()
  genConfig.sid = 99
  genConfig.speed = 1.0
  genConfig.silenceScale = 0.2

  let audio = tts.generateWithConfig(
    text: text, config: genConfig, callback: callback, arg: arg)
  let filename = "test-vits-en.wav"
  let ok = audio.save(filename: filename)
  if ok == 1 {
    print("\nSaved to:\(filename)")
  } else {
    print("Failed to save to \(filename)")
  }
}

@main
struct App {
  static func main() {
    run()
  }
}
