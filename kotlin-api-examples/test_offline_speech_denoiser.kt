package com.k2fsa.sherpa.onnx
// Please download test files in this script from
// https://github.com/k2-fsa/sherpa-onnx/releases/tag/speech-enhancement-models

fun main() {
  test()
}

fun test() {
  val denoiser  = createOfflineSpeechDenoiser()

  val waveFilename = "./inp_16k.wav";

  val waveData = WaveReader.readWaveFromFile(
      filename = waveFilename,
  )

  val denoised = denoiser.run(waveData.samples, waveData.sampleRate);
  denoised.save(filename="./enhanced-16k.wav")
  println("saved to ./enhanced-16k.wav")
}

fun createOfflineSpeechDenoiser(): OfflineSpeechDenoiser {
  val config = OfflineSpeechDenoiserConfig(
      model = OfflineSpeechDenoiserModelConfig(
        gtcrn = OfflineSpeechDenoiserGtcrnModelConfig(
          model = "./gtcrn_simple.onnx"
        ),
        provider = "cpu",
        numThreads = 1,
      ),
  )

  println(config)

  return OfflineSpeechDenoiser(config = config)
}


