package com.k2fsa.sherpa.onnx

fun main() {
  val recognizer = createOfflineRecognizer()
  val waveFilename = "./sherpa-onnx-cohere-transcribe-14-lang-int8-2026-04-01/test_wavs/en.wav"

  val waveData = WaveReader.readWaveFromFile(
      filename = waveFilename,
  )

  val stream = recognizer.createStream()
  stream.setOption("language", "en")
  stream.acceptWaveform(waveData.samples, sampleRate=waveData.sampleRate)

  val start = System.currentTimeMillis()
  recognizer.decode(stream)
  val stop = System.currentTimeMillis()

  val result = recognizer.getResult(stream)
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
  val modelDir = "./sherpa-onnx-cohere-transcribe-14-lang-int8-2026-04-01"
  val modelConfig = OfflineModelConfig(
      cohereTranscribe = OfflineCohereTranscribeModelConfig(
          encoder = "$modelDir/encoder.int8.onnx",
          decoder = "$modelDir/decoder.int8.onnx",
          usePunct = true,
          useItn = true,
      ),
      tokens = "$modelDir/tokens.txt",
      numThreads = 2,
      debug = false,
  )

  val config = OfflineRecognizerConfig(
      modelConfig = modelConfig,
  )

  return OfflineRecognizer(config = config)
}
