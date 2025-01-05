class MyClass {
  func playSamples(samples: [Float]) {
    print("Play \(samples.count) samples")
  }
}

func run() {
  let acousticModel = "./matcha-icefall-zh-baker/model-steps-3.onnx"
  let vocoder = "./hifigan_v2.onnx"
  let lexicon = "./matcha-icefall-zh-baker/lexicon.txt"
  let tokens = "./matcha-icefall-zh-baker/tokens.txt"
  let dictDir = "./matcha-icefall-zh-baker/dict"
  let ruleFsts =
    "./matcha-icefall-zh-baker/phone.fst,./matcha-icefall-zh-baker/date.fst,./matcha-icefall-zh-baker/number.fst"
  let matcha = sherpaOnnxOfflineTtsMatchaModelConfig(
    acousticModel: acousticModel,
    vocoder: vocoder,
    lexicon: lexicon,
    tokens: tokens,
    dictDir: dictDir
  )
  let modelConfig = sherpaOnnxOfflineTtsModelConfig(matcha: matcha, debug: 0)
  var ttsConfig = sherpaOnnxOfflineTtsConfig(model: modelConfig, ruleFsts: ruleFsts)

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

  let text = "某某银行的副行长和一些行政领导表示，他们去过长江和长白山; 经济不断增长。2024年12月31号，拨打110或者18920240511。123456块钱。"
  let sid = 0
  let speed: Float = 1.0

  let audio = tts.generateWithCallbackWithArg(
    text: text, callback: callback, arg: arg, sid: sid, speed: speed)
  let filename = "test-matcha-zh.wav"
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
