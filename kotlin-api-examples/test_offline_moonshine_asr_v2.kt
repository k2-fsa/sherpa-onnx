package com.k2fsa.sherpa.onnx

fun main() {
  val recognizer = createOfflineRecognizer()
  val waveFilename = "./sherpa-onnx-moonshine-tiny-en-quantized-2026-02-27/test_wavs/0.wav"

  val waveData = WaveReader.readWaveFromFile(
      filename = waveFilename,
  )

  var stream = recognizer.createStream()
  stream.acceptWaveform(waveData.samples, sampleRate=waveData.sampleRate)
  recognizer.decode(stream)

  var result = recognizer.getResult(stream)
  println(result)

  stream.release()
  recognizer.release()
}


fun createOfflineRecognizer(): OfflineRecognizer {
  val config = OfflineRecognizerConfig(
      modelConfig = getOfflineModelConfig(type = 53)!!,
  )

  return OfflineRecognizer(config = config)
}
