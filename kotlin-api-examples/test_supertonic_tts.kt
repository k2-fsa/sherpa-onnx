package com.k2fsa.sherpa.onnx

fun main() {
  testSupertonicTts()
}

fun testSupertonicTts() {
  // see https://github.com/k2-fsa/sherpa-onnx/releases/tag/tts-models
  val modelDir = "./sherpa-onnx-supertonic-tts-int8-2026-03-06"
  val config = OfflineTtsConfig(
    model=OfflineTtsModelConfig(
      supertonic=OfflineTtsSupertonicModelConfig(
        durationPredictor="$modelDir/duration_predictor.int8.onnx",
        textEncoder="$modelDir/text_encoder.int8.onnx",
        vectorEstimator="$modelDir/vector_estimator.int8.onnx",
        vocoder="$modelDir/vocoder.int8.onnx",
        ttsJson="$modelDir/tts.json",
        unicodeIndexer="$modelDir/unicode_indexer.bin",
        voiceStyle="$modelDir/voice.bin",
      ),
      numThreads=2,
      debug=true,
    ),
  )
  val tts = OfflineTts(config=config)

  val genConfig = GenerationConfig(
    sid = 6,
    speed = 1.25f,
    numSteps = 5,
    extra = mapOf(
        "lang" to "en",
    )
  )

  val text = "Today as always, men fall into two groups: slaves and free men. Whoever does not have two-thirds of his day for himself, is a slave, whatever he may be, a statesman, a businessman, an official, or a scholar."

  val audio = tts.generateWithConfigAndCallback(text=text, config=genConfig, callback=::supertonicCallback)
  audio.save(filename="test-supertonic-en.wav")
  tts.release()
  println("Saved to test-supertonic-en.wav")
}

fun supertonicCallback(samples: FloatArray): Int {
  println("callback got called with ${samples.size} samples")

  // 1 means to continue
  // 0 means to stop
  return 1
}
