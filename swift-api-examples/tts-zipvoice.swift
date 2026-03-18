import Foundation

class ZipVoiceTtsProgressHandler {
  func progress(samples: [Float], progress: Float) {
    print(String(format: "Received %d samples, Progress: %.2f%%", samples.count, progress * 100))
  }
}

func runZipVoiceTtsDemo() {
  let zipvoice = sherpaOnnxOfflineTtsZipvoiceModelConfig(
    tokens: "./sherpa-onnx-zipvoice-distill-int8-zh-en-emilia/tokens.txt",
    encoder: "./sherpa-onnx-zipvoice-distill-int8-zh-en-emilia/encoder.int8.onnx",
    decoder: "./sherpa-onnx-zipvoice-distill-int8-zh-en-emilia/decoder.int8.onnx",
    vocoder: "./vocos_24khz.onnx",
    dataDir: "./sherpa-onnx-zipvoice-distill-int8-zh-en-emilia/espeak-ng-data",
    lexicon: "./sherpa-onnx-zipvoice-distill-int8-zh-en-emilia/lexicon.txt"
  )

  let modelConfig = sherpaOnnxOfflineTtsModelConfig(numThreads: 2, zipvoice: zipvoice)
  var ttsConfig = sherpaOnnxOfflineTtsConfig(model: modelConfig)
  ttsConfig.model.debug = 1

  let tts = SherpaOnnxOfflineTtsWrapper(config: &ttsConfig)

  let referenceAudioFile = "./sherpa-onnx-zipvoice-distill-int8-zh-en-emilia/test_wavs/leijun-1.wav"
  let referenceWave = SherpaOnnxWaveWrapper.readWave(filename: referenceAudioFile)

  var genConfig = SherpaOnnxGenerationConfigSwift()
  genConfig.speed = 1.0
  genConfig.referenceAudio = referenceWave.samples
  genConfig.referenceSampleRate = referenceWave.sampleRate
  genConfig.referenceText = "那还是三十六年前, 一九八七年. 我呢考上了武汉大学的计算机系."
  genConfig.numSteps = 4
  genConfig.extra = ["min_char_in_sentence": "10"]

  let text = "小米的价值观是真诚, 热爱. 真诚，就是不欺人也不自欺. 热爱, 就是全心投入并享受其中."

  func generateAndSave(
    outputFile: String, callback: TtsProgressCallbackWithArg? = nil,
    arg: UnsafeMutableRawPointer? = nil
  ) {
    let audio = tts.generateWithConfig(
      text: text,
      config: genConfig,
      callback: callback,
      arg: arg
    )

    if audio.save(filename: outputFile) == 1 {
      print("Saved to: \(outputFile)")
    } else {
      print("Failed to save to \(outputFile)")
    }
  }

  let useCallback = true
  if useCallback {
    let progressHandler = ZipVoiceTtsProgressHandler()
    let arg = Unmanaged.passUnretained(progressHandler).toOpaque()

    let callback: TtsProgressCallbackWithArg = { samples, n, progress, arg in
      let handler = Unmanaged<ZipVoiceTtsProgressHandler>.fromOpaque(arg!).takeUnretainedValue()

      let buffer: [Float] =
        samples != nil ? Array(UnsafeBufferPointer(start: samples, count: Int(n))) : []
      handler.progress(samples: buffer, progress: progress)
      return 1
    }

    generateAndSave(outputFile: "generated-zipvoice-callback.wav", callback: callback, arg: arg)
  } else {
    generateAndSave(outputFile: "generated-zipvoice-direct.wav")
  }
}

@main
struct App {
  static func main() {
    runZipVoiceTtsDemo()
  }
}
