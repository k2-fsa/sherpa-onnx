package com.k2fsa.sherpa.onnx

fun main() {
  val types = arrayOf(0, 2, 5, 6)
  for (type in types) {
    test(type)
  }
}

fun test(type: Int) {
  val recognizer = createOfflineRecognizer(type)

  val waveFilename = when (type) {
    0 -> "./sherpa-onnx-paraformer-zh-2023-03-28/test_wavs/0.wav"
    2 -> "./sherpa-onnx-whisper-tiny.en/test_wavs/0.wav"
    5 -> "./sherpa-onnx-zipformer-multi-zh-hans-2023-9-2/test_wavs/1.wav"
    6 -> "./sherpa-onnx-nemo-ctc-en-citrinet-512/test_wavs/8k.wav"
    else -> null
  }

  val objArray = WaveReader.readWaveFromFile(
      filename = waveFilename!!,
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

fun createOfflineRecognizer(type: Int): OfflineRecognizer {
  val config = OfflineRecognizerConfig(
      featConfig = getFeatureConfig(sampleRate = 16000, featureDim = 80),
      modelConfig = getOfflineModelConfig(type = type)!!,
  )

  return OfflineRecognizer(config = config)
}
