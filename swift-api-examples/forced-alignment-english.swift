import Foundation

func run() {
  let modelDir = "./english-mfa-v3.1.0"
  let wav = "./english-mfa-v3.1.0/en.wav"
  // The transcript must match the wav.
  let text =
    "ask not what your country can do for you, " +
    "ask what you can do for your country."

  var config = ForcedAlignerConfig()
  config.modelDir = modelDir
  config.lexicon = modelDir + "/lexicon.txt"

  guard let aligner = ForcedAligner(config: config) else {
    print("Failed to create the forced aligner")
    return
  }
  let stream = aligner.createStream()

  // Optional: per-stream options, e.g. debug output of one utterance.
  stream.setOption(key: "debug", value: "1")

  let wave = SherpaOnnxWaveWrapper.readWave(filename: wav)
  stream.acceptWaveform(samples: wave.samples, sampleRate: wave.sampleRate)

  let start = Date()
  guard let result = aligner.compute(stream: stream, text: text) else {
    print("Alignment failed")
    return
  }
  let elapsed = Date().timeIntervalSince(start)

  // tiers[0] is the "phones" tier; every interval has a phone symbol.
  print("phones:")
  for iv in result.tiers[0].intervals {
    print("\(String(format: "%.3f - %.3f", iv.start, iv.end))  \(iv.text)")
  }

  // tiers[1] is the "words" tier; an empty text marks a silence gap.
  print("\nwords:")
  for iv in result.tiers[1].intervals {
    let label = iv.text.isEmpty ? "<silence>" : iv.text
    print("\(String(format: "%.3f - %.3f", iv.start, iv.end))  \(label)")
  }

  let output = "en.TextGrid"
  try! result.textGrid.write(toFile: output, atomically: true, encoding: .utf8)
  print("\nSaved to \(output)")

  let duration = Float(wave.samples.count) / Float(wave.sampleRate)
  print("Elapsed seconds: \(String(format: "%.3f", elapsed))")
  print("Audio duration in seconds: \(String(format: "%.3f", duration))")
  print("RTF: \(String(format: "%.3f/%.3f = %.3f", elapsed, duration, elapsed / Double(duration)))")
}

@main
struct App {
  static func main() {
    run()
  }
}
