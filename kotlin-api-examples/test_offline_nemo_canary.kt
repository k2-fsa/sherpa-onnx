package com.k2fsa.sherpa.onnx

fun main() {
  val recognizer = createOfflineRecognizer()
  val waveFilename = "./sherpa-onnx-nemo-canary-180m-flash-en-es-de-fr-int8/test_wavs/en.wav"

  val objArray = WaveReader.readWaveFromFile(
      filename = waveFilename,
  )
  val samples: FloatArray = objArray[0] as FloatArray
  val sampleRate: Int = objArray[1] as Int

  var stream = recognizer.createStream()
  stream.acceptWaveform(samples, sampleRate=sampleRate)
  recognizer.decode(stream)

  var result = recognizer.getResult(stream)
  println("English: $result")

  stream.release()

  // now output text in German
  val config = recognizer.config.copy(modelConfig=recognizer.config.modelConfig.copy(
    canary=recognizer.config.modelConfig.canary.copy(
      tgtLang="de"
    )
  ))
  recognizer.setConfig(config)

  stream = recognizer.createStream()
  stream.acceptWaveform(samples, sampleRate=sampleRate)
  recognizer.decode(stream)

  result = recognizer.getResult(stream)
  println("German: $result")

  stream.release()
  recognizer.release()
}


fun createOfflineRecognizer(): OfflineRecognizer {
  val config = OfflineRecognizerConfig(
      modelConfig = getOfflineModelConfig(type = 32)!!,
  )

  return OfflineRecognizer(config = config)
}
