package com.k2fsa.sherpa.onnx
// Please download test files in this script from
// https://github.com/k2-fsa/sherpa-onnx/releases/tag/speech-enhancement-models

fun main() {
  val denoiser = createOfflineSpeechDenoiserDpdfNet()
  val waveData = WaveReader.readWaveFromFile(filename = "./inp_16k.wav")
  val denoised = denoiser.run(waveData.samples, waveData.sampleRate)
  denoised.save(filename = "./enhanced-dpdfnet-16k.wav")
  println("saved to ./enhanced-dpdfnet-16k.wav")
}

fun createOfflineSpeechDenoiserDpdfNet(): OfflineSpeechDenoiser {
  val config = OfflineSpeechDenoiserConfig(
      model = OfflineSpeechDenoiserModelConfig(
        dpdfnet = OfflineSpeechDenoiserDpdfNetModelConfig(
          model = "./dpdfnet_baseline.onnx"
        ),
        provider = "cpu",
        numThreads = 1,
      ),
  )

  return OfflineSpeechDenoiser(config = config)
}
