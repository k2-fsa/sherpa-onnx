// This file shows the piper-phonemize text frontend for TTS.
//
// The text is phonemized outside of sherpa-onnx with piper-phonemize
// (https://github.com/csukuangfj/piper-phonemize) and the resulting phoneme
// codepoints are passed via GenerationConfig.phonemeCodepoints. No lexicon
// file and no espeak-ng-data are needed.
//
// If you want to use a lexicon instead, please see test_tts_lexicon.kt.
package com.k2fsa.sherpa.onnx

import com.github.csukuangfj.piper.phonemize.PiperPhonemize

fun main() {
  testPiperEnPhonemize()
  testKittenEnPhonemize()
  testKokoroEnPhonemize()
  testVitsDePhonemize()
}

// Phonemize the text and return the phoneme codepoints, one inner array per
// sentence. The espeak-ng voice is taken from the model metadata; fall back
// to [fallbackVoice] if the model has no language information.
fun phonemize(tts: OfflineTts, text: String, fallbackVoice: String): Array<IntArray> {
  var voice = tts.lang()
  if (voice.isEmpty()) {
    println("Warning: the model has no language information. Falling back to $fallbackVoice.")
    voice = fallbackVoice
  }
  println("Phonemizer language: $voice")

  val result = PiperPhonemize.text(text, voice)
    ?: throw RuntimeException("Failed to phonemize the text")
  result.use {
    return Array(it.numSentences) { i -> it.getPhonemes(i) }
  }
}

fun testPiperEnPhonemize() {
  // see https://github.com/k2-fsa/sherpa-onnx/releases/tag/tts-models
  // https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/vits-piper-en_US-amy-low.tar.bz2
  PiperPhonemize.initialize()
  var config = OfflineTtsConfig(
    model=OfflineTtsModelConfig(
      vits=OfflineTtsVitsModelConfig(
        model="./vits-piper-en_US-amy-low/en_US-amy-low.onnx",
        tokens="./vits-piper-en_US-amy-low/tokens.txt",
      ),
      numThreads=1,
      debug=true,
    )
  )
  val tts = OfflineTts(config=config)
  val text = "“Today as always, men fall into two groups: slaves and free men. Whoever does not have two-thirds of his day for himself, is a slave, whatever he may be: a statesman, a businessman, an official, or a scholar.”"
  val genConfig = GenerationConfig(
    silenceScale = 0.2f,
    phonemeCodepoints = phonemize(tts, text, "en-us"),
  )
  // The text has already been phonemized; pass an empty string here.
  val audio = tts.generateWithConfigAndCallback(text="", config=genConfig, callback=::callback)
  audio.save(filename="test-piper-en-phonemize.wav")
  tts.release()
  println("Saved to test-piper-en-phonemize.wav")
}

fun testKittenEnPhonemize() {
  // see https://github.com/k2-fsa/sherpa-onnx/releases/tag/tts-models
  PiperPhonemize.initialize()
  var config = OfflineTtsConfig(
    model=OfflineTtsModelConfig(
      kitten=OfflineTtsKittenModelConfig(
        model="./kitten-nano-en-v0_1-fp16/model.fp16.onnx",
        voices="./kitten-nano-en-v0_1-fp16/voices.bin",
        tokens="./kitten-nano-en-v0_1-fp16/tokens.txt",
      ),
      numThreads=2,
      debug=true,
    ),
  )
  val tts = OfflineTts(config=config)
  val text = "How are you doing today?"
  val genConfig = GenerationConfig(
    sid = 7,
    silenceScale = 0.2f,
    phonemeCodepoints = phonemize(tts, text, "en-us"),
  )
  val audio = tts.generateWithConfigAndCallback(text="", config=genConfig, callback=::callback)
  audio.save(filename="test-kitten-en-phonemize.wav")
  tts.release()
  println("Saved to test-kitten-en-phonemize.wav")
}

fun testKokoroEnPhonemize() {
  // see https://github.com/k2-fsa/sherpa-onnx/releases/tag/tts-models
  PiperPhonemize.initialize()
  var config = OfflineTtsConfig(
    model=OfflineTtsModelConfig(
      kokoro=OfflineTtsKokoroModelConfig(
        model="./kokoro-en-v0_19/model.onnx",
        voices="./kokoro-en-v0_19/voices.bin",
        tokens="./kokoro-en-v0_19/tokens.txt",
      ),
      numThreads=2,
      debug=true,
    ),
  )
  val tts = OfflineTts(config=config)
  val text = "How are you doing today?"
  val genConfig = GenerationConfig(
    silenceScale = 0.2f,
    phonemeCodepoints = phonemize(tts, text, "en-us"),
  )
  val audio = tts.generateWithConfigAndCallback(text="", config=genConfig, callback=::callback)
  audio.save(filename="test-kokoro-en-phonemize.wav")
  tts.release()
  println("Saved to test-kokoro-en-phonemize.wav")
}

fun testVitsDePhonemize() {
  // see https://github.com/k2-fsa/sherpa-onnx/releases/tag/tts-models
  // https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/vits-piper-de_DE-glados-low.tar.bz2
  //
  // There is no German lexicon, so this model is phonemize-only.
  PiperPhonemize.initialize()
  var config = OfflineTtsConfig(
    model=OfflineTtsModelConfig(
      vits=OfflineTtsVitsModelConfig(
        model="./vits-piper-de_DE-glados-low/de_DE-glados-low.onnx",
        tokens="./vits-piper-de_DE-glados-low/tokens.txt",
      ),
      numThreads=1,
      debug=true,
    )
  )
  val tts = OfflineTts(config=config)
  val text = "Alles hat ein Ende, nur die Wurst hat zwei."
  val genConfig = GenerationConfig(
    silenceScale = 0.2f,
    phonemeCodepoints = phonemize(tts, text, "de"),
  )
  val audio = tts.generateWithConfigAndCallback(text="", config=genConfig, callback=::callback)
  audio.save(filename="test-vits-de-phonemize.wav")
  tts.release()
  println("Saved to test-vits-de-phonemize.wav")
}

fun callback(samples: FloatArray): Int {
  println("callback got called with ${samples.size} samples");

  // 1 means to continue
  // 0 means to stop
  return 1
}
