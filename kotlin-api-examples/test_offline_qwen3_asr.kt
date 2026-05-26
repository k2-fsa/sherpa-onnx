package com.k2fsa.sherpa.onnx

fun main() {
  val recognizer = createOfflineRecognizer()
  val waveFilename = "./sherpa-onnx-qwen3-asr-0.6B-int8-2026-03-25/test_wavs/raokouling.wav"

  val waveData = WaveReader.readWaveFromFile(
      filename = waveFilename,
  )

  var stream = recognizer.createStream()
  stream.acceptWaveform(waveData.samples, sampleRate=waveData.sampleRate)

  val start = System.currentTimeMillis()
  recognizer.decode(stream)
  val stop = System.currentTimeMillis()

  var result = recognizer.getResult(stream)
  println(result)

  val timeElapsedSeconds = (stop - start) / 1000.0f
  val audioDuration = waveData.samples.size / waveData.sampleRate.toFloat()
  val realTimeFactor = timeElapsedSeconds / audioDuration

  println(String.format("-- elapsed : %.3f seconds", timeElapsedSeconds))
  println(String.format("-- audio duration: %.3f seconds", audioDuration))
  println(String.format("-- real-time factor (RTF): %.3f", realTimeFactor))

  stream.release()
  recognizer.release()
}


fun createOfflineRecognizer(): OfflineRecognizer {
  val config = OfflineRecognizerConfig(
      modelConfig = getOfflineModelConfig(type = 61)!!,
  )

  return OfflineRecognizer(config = config)
}
