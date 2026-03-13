package com.k2fsa.sherpa.onnx

fun main() {
  val types = arrayOf(0, 2, 5, 6, 15, 21, 24, 25, 31)
  for (type in types) {
    test(type)
  }
}

fun test(type: Int) {
  val recognizer = createOfflineRecognizer(type)

  val waveFilename = when (type) {
    0 -> "./sherpa-onnx-paraformer-zh-2023-09-14/test_wavs/0.wav"
    2 -> "./sherpa-onnx-whisper-tiny.en/test_wavs/0.wav"
    5 -> "./sherpa-onnx-zipformer-multi-zh-hans-2023-9-2/test_wavs/1.wav"
    6 -> "./sherpa-onnx-nemo-ctc-en-citrinet-512/test_wavs/8k.wav"
    15 -> "./sherpa-onnx-sense-voice-zh-en-ja-ko-yue-int8-2024-07-17/test_wavs/zh.wav"
    21 -> "./sherpa-onnx-moonshine-tiny-en-int8/test_wavs/0.wav"
    24 -> "./sherpa-onnx-fire-red-asr-large-zh_en-2025-02-16/test_wavs/0.wav"
    25 -> "./sherpa-onnx-dolphin-base-ctc-multi-lang-int8-2025-04-02/test_wavs/0.wav"
    31 -> "./sherpa-onnx-zipformer-ctc-zh-int8-2025-07-03/test_wavs/0.wav"
    else -> null
  }

  val waveData = WaveReader.readWaveFromFile(
      filename = waveFilename!!,
  )

  val stream = recognizer.createStream()
  stream.acceptWaveform(waveData.samples, sampleRate=waveData.sampleRate)
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
