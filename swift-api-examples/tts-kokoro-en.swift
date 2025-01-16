class MyClass {
  func playSamples(samples: [Float]) {
    print("Play \(samples.count) samples")
  }
}

func run() {
  let model = "./kokoro-en-v0_19/model.onnx"
  let voices = "./kokoro-en-v0_19/voices.bin"
  let tokens = "./kokoro-en-v0_19/tokens.txt"
  let dataDir = "./kokoro-en-v0_19/espeak-ng-data"
  let kokoro = sherpaOnnxOfflineTtsKokoroModelConfig(
    model: model,
    voices: voices,
    tokens: tokens,
    dataDir: dataDir
  )
  let modelConfig = sherpaOnnxOfflineTtsModelConfig(kokoro: kokoro, debug: 0)
  var ttsConfig = sherpaOnnxOfflineTtsConfig(model: modelConfig)

  let myClass = MyClass()

  // We use Unretained here so myClass must be kept alive as the callback is invoked
  //
  // See also
  // https://medium.com/codex/swift-c-callback-interoperability-6d57da6c8ee6
  let arg = Unmanaged<MyClass>.passUnretained(myClass).toOpaque()

  let callback: TtsCallbackWithArg = { samples, n, arg in
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
    "Friends fell out often because life was changing so fast. The easiest thing in the world was to lose touch with someone."
  let sid = 0
  let speed: Float = 1.0

  let audio = tts.generateWithCallbackWithArg(
    text: text, callback: callback, arg: arg, sid: sid, speed: speed)
  let filename = "test-kokoro-en.wav"
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
