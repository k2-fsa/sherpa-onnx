package com.k2fsa.sherpa.onnx

fun main() {
  val recognizer = createOfflineRecognizer()
  val waveFilename = "./sherpa-onnx-funasr-nano-int8-2025-12-30/test_wavs/lyrics.wav"

  val objArray = WaveReader.readWaveFromFile(
      filename = waveFilename,
  )
  val samples: FloatArray = objArray[0] as FloatArray
  val sampleRate: Int = objArray[1] as Int

  var stream = recognizer.createStream()
  stream.acceptWaveform(samples, sampleRate=sampleRate)
  recognizer.decode(stream)

  var result = recognizer.getResult(stream)
  println(result)

  stream.release()
  recognizer.release()
}


fun createOfflineRecognizer(): OfflineRecognizer {
  val config = OfflineRecognizerConfig(
      modelConfig = getOfflineModelConfig(type = 46)!!,
  )

  return OfflineRecognizer(config = config)
}
