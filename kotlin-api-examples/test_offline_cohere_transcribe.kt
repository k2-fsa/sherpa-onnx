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
  recognizer.decode(stream)

  val result = recognizer.getResult(stream)
  println(result)

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
      debug = true,
  )

  val config = OfflineRecognizerConfig(
      modelConfig = modelConfig,
  )

  return OfflineRecognizer(config = config)
}
