package com.k2fsa.sherpa.onnx

fun main() {
  testPocketTts()
}

// Builds the config through getOfflineTtsConfig so the Pocket branch of that
// helper is exercised end to end, not just the raw data class.
fun testPocketTts() {
  // see https://github.com/k2-fsa/sherpa-onnx/releases/tag/tts-models
  val modelDir = "./sherpa-onnx-pocket-tts-int8-2026-01-26"

  val config = getOfflineTtsConfig(
    modelDir = modelDir,
    isPocket = true,
    lmFlow = "lm_flow.int8.onnx",
    lmMain = "lm_main.int8.onnx",
    pocketEncoder = "encoder.onnx",
    pocketDecoder = "decoder.int8.onnx",
    textConditioner = "text_conditioner.onnx",
    vocabJson = "vocab.json",
    tokenScoresJson = "token_scores.json",
  )
  val tts = OfflineTts(config = config)

  val referenceAudioFilename = "$modelDir/test_wavs/bria.wav"
  val wave = WaveReader.readWave(
      filename = referenceAudioFilename,
  )

  val genConfig = GenerationConfig(
    referenceAudio = wave.samples,
    referenceSampleRate = wave.sampleRate,
    numSteps = 5,
    extra = mapOf(
        "temperature" to "0.7",
        "chunk_size" to "15",
    )
  )

  val text = "Today as always, men fall into two groups: slaves and free men. Whoever does not have two-thirds of his day for himself, is a slave, whatever he may be, a statesman, a businessman, an official, or a scholar."

  val audio = tts.generateWithConfigAndCallback(text=text, config=genConfig, callback=::callback)
  if (audio.samples.isEmpty()) {
    println("FAIL - PocketTTS produced no audio")
    System.exit(1)
  }
  audio.save(filename="out-bria.wav")
  tts.release()
  println("Saved to out-bria.wav")
}

fun callback(samples: FloatArray): Int {
  println("callback got called with ${samples.size} samples")

  // 1 means to continue
  // 0 means to stop
  return 1
}
