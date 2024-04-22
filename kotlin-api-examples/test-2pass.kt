package com.k2fsa.sherpa.onnx

fun main() {
  test2Pass()
}

fun test2Pass() {
  val firstPass = createFirstPass()
  val secondPass = createSecondPass()

  val waveFilename = "./sherpa-onnx-streaming-zipformer-en-20M-2023-02-17/test_wavs/0.wav"

  var objArray = WaveReader.readWaveFromFile(
      filename = waveFilename,
  )
  var samples: FloatArray = objArray[0] as FloatArray
  var sampleRate: Int = objArray[1] as Int

  firstPass.acceptWaveform(samples, sampleRate = sampleRate)
  while (firstPass.isReady()) {
      firstPass.decode()
  }

  var text = firstPass.text
  println("First pass text: $text")

  val stream = secondPass.createStream()
  stream.acceptWaveform(samples, sampleRate=sampleRate)
  secondPass.decode(stream)

  val result = secondPass.getResult(stream)
  println(result)

  stream.release()
  secondPass.release()
}

fun createFirstPass(): SherpaOnnx {
  val config = OnlineRecognizerConfig(
      featConfig = getFeatureConfig(sampleRate = 16000, featureDim = 80),
      modelConfig = getModelConfig(type = 1)!!,
      endpointConfig = getEndpointConfig(),
      enableEndpoint = true,
  )

  return SherpaOnnx(config = config)
}

fun createSecondPass(): OfflineRecognizer {
  val config = OfflineRecognizerConfig(
      featConfig = getFeatureConfig(sampleRate = 16000, featureDim = 80),
      modelConfig = getOfflineModelConfig(type = 2)!!,
  )

  return OfflineRecognizer(config = config)
}
