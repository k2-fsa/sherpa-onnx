package com.k2fsa.sherpa.onnx

fun main() {
  val recognizer = createOfflineRecognizer()
  val waveFilename = "./test-hr.wav"

  val waveData = WaveReader.readWaveFromFile(
      filename = waveFilename,
  )

  val stream = recognizer.createStream()
  stream.acceptWaveform(waveData.samples, sampleRate=waveData.sampleRate)
  recognizer.decode(stream)

  val result = recognizer.getResult(stream)
  println(result)

  stream.release()
  recognizer.release()
}

fun createOfflineRecognizer(): OfflineRecognizer {
  val config = OfflineRecognizerConfig(
      featConfig = getFeatureConfig(sampleRate = 16000, featureDim = 80),
      modelConfig = getOfflineModelConfig(type = 15)!!,
      hr = HomophoneReplacerConfig(
        lexicon = "./lexicon.txt",
        ruleFsts = "./replace.fst"),
  )

  return OfflineRecognizer(config = config)
}
