package com.k2fsa.sherpa.onnx

fun main() {
  test()
}

fun test() {
  val recognizer = createOfflineRecognizer()
  val waveFilename = "./itn-zh-number.wav";

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
      modelConfig = getOfflineModelConfig(0)!!,
      ruleFsts = "./itn_zh_number.fst",
  )

  return OfflineRecognizer(config = config)
}

