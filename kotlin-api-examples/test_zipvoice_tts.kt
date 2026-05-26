package com.k2fsa.sherpa.onnx

fun main() {
  testZipVoiceTts()
}

fun testZipVoiceTts() {
  val modelDir = "./sherpa-onnx-zipvoice-distill-int8-zh-en-emilia"
  val referenceAudioFilename = "$modelDir/test_wavs/leijun-1.wav"
  val wave = WaveReader.readWave(filename = referenceAudioFilename)

  val config = OfflineTtsConfig(
    model = OfflineTtsModelConfig(
      zipvoice = OfflineTtsZipVoiceModelConfig(
        tokens = "$modelDir/tokens.txt",
        encoder = "$modelDir/encoder.int8.onnx",
        decoder = "$modelDir/decoder.int8.onnx",
        vocoder = "./vocos_24khz.onnx",
        dataDir = "$modelDir/espeak-ng-data",
        lexicon = "$modelDir/lexicon.txt",
      ),
      numThreads = 2,
      debug = false,
    ),
  )

  val tts = OfflineTts(config = config)
  val text = "小米的价值观是真诚, 热爱. 真诚，就是不欺人也不自欺. 热爱, 就是全心投入并享受其中."
  val referenceText = "那还是三十六年前, 一九八七年. 我呢考上了武汉大学的计算机系."
  val genConfig = GenerationConfig(
    referenceAudio = wave.samples,
    referenceSampleRate = wave.sampleRate,
    referenceText = referenceText,
    numSteps = 4,
    extra = mapOf("min_char_in_sentence" to "10"),
  )

  val audio = tts.generateWithConfigAndCallback(text = text, config = genConfig, callback = ::callback)
  audio.save(filename = "test-zipvoice-zh-en.wav")
  tts.release()
  println("Saved to test-zipvoice-zh-en.wav")
}

fun callback(samples: FloatArray): Int {
  println("callback got called with ${samples.size} samples")

  // 1 means to continue
  // 0 means to stop
  return 1
}
