class MyClass {
  func playSamples(samples: [Float]) {
    print("Play \(samples.count) samples")
  }
}

func run() {
  let lmFlow = "./sherpa-onnx-pocket-tts-int8-2026-01-26/lm_flow.int8.onnx"
  let lmMain = "./sherpa-onnx-pocket-tts-int8-2026-01-26/lm_main.int8.onnx"
  let encoder = "./sherpa-onnx-pocket-tts-int8-2026-01-26/encoder.onnx"
  let decoder = "./sherpa-onnx-pocket-tts-int8-2026-01-26/decoder.int8.onnx"
  let textConditioner = "./sherpa-onnx-pocket-tts-int8-2026-01-26/text_conditioner.onnx"

  let pocket = sherpaOnnxOfflineTtsPocketModelConfig(
    lmFlow: lmFlow,
    lmMain: lmMain,
    encoder: encoder,
    decoder: decoder,
    textConditioner: textConditioner
  )
  let modelConfig = sherpaOnnxOfflineTtsModelConfig(debug: 0, pocket: pocket)
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
  let filename = "test-kitten-en.wav"
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
