package com.k2fsa.sherpa.onnx

fun main() {
  val recognizer = createOfflineRecognizer()

  val waveFilename = "./sherpa-onnx-streaming-zipformer-en-20M-2023-02-17/test_wavs/0.wav"

  val objArray = WaveReader.readWaveFromFile(
      filename = waveFilename,
  )
  val samples: FloatArray = objArray[0] as FloatArray
  val sampleRate: Int = objArray[1] as Int

  val stream = recognizer.createStream()
  stream.acceptWaveform(samples, sampleRate=sampleRate)
  recognizer.decode(stream)

  val result = recognizer.getResult(stream)
  println(result)

  stream.release()
  recognizer.release()
}

fun createOfflineRecognizer(): OfflineRecognizer {
  val config = OfflineRecognizerConfig(
      featConfig = getFeatureConfig(sampleRate = 16000, featureDim = 80),
      modelConfig = getOfflineModelConfig(type = 2)!!,
  )

  return OfflineRecognizer(config = config)
}
