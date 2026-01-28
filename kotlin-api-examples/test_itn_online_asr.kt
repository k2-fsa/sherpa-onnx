package com.k2fsa.sherpa.onnx

fun main() {
  test()
}

fun test() {
  val recognizer = createOnlineRecognizer()
  val waveFilename = "./itn-zh-number.wav";

  val waveData = WaveReader.readWaveFromFile(
      filename = waveFilename,
  )

  val stream = recognizer.createStream()
  stream.acceptWaveform(waveData.samples, sampleRate=waveData.sampleRate)
  while (recognizer.isReady(stream)) {
    recognizer.decode(stream)
  }

  val result = recognizer.getResult(stream).text
  println(result)

  stream.release()
  recognizer.release()
}

fun createOnlineRecognizer(): OnlineRecognizer {
  val config = OnlineRecognizerConfig(
      featConfig = getFeatureConfig(sampleRate = 16000, featureDim = 80),
      modelConfig = getModelConfig(8)!!,
  )

  config.ruleFsts = "./itn_zh_number.fst"
  println(config)

  return OnlineRecognizer(config = config)
}

