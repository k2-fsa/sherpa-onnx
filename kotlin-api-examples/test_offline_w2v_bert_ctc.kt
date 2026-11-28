package com.k2fsa.sherpa.onnx

fun main() {
  val recognizer = createOfflineRecognizer()
  val waveFilename = "./sherpa-onnx-w2v-bert-ctc-gl-int8-2026-11-20/test_wavs/0.wav"

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
  val modelDir = "./sherpa-onnx-w2v-bert-ctc-gl-int8-2026-11-20"
  val modelConfig = OfflineModelConfig(
      w2vBertCtc = OfflineW2vBertCtcModelConfig(
          model = "$modelDir/model.int8.onnx",
      ),
      tokens = "$modelDir/tokens.txt",
  )

  val config = OfflineRecognizerConfig(
      modelConfig = modelConfig,
  )

  return OfflineRecognizer(config = config)
}
