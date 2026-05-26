package com.k2fsa.sherpa.onnx

// Please download test files in this script from
// https://github.com/k2-fsa/sherpa-onnx/releases/tag/speech-enhancement-models

fun main() {
  testGtcrn()
  testDpdfNet()
}

fun testGtcrn() {
  val denoiser = createOnlineSpeechDenoiserGtcrn()
  val waveData = WaveReader.readWaveFromFile("./inp_16k.wav")
  val output = mutableListOf<Float>()
  val frameShift = denoiser.frameShiftInSamples

  var start = 0
  while (start < waveData.samples.size) {
    val end = minOf(start + frameShift, waveData.samples.size)
    val chunk = waveData.samples.copyOfRange(start, end)
    val denoised = denoiser.run(chunk, waveData.sampleRate)
    output.addAll(denoised.samples.asList())
    start = end
  }

  output.addAll(denoiser.flush().samples.asList())
  DenoisedAudio(output.toFloatArray(), denoiser.sampleRate).save(
    filename = "./enhanced-online-gtcrn.wav"
  )
  println("saved to ./enhanced-online-gtcrn.wav")

  denoiser.release()
}

fun testDpdfNet() {
  val denoiser = createOnlineSpeechDenoiserDpdfNet()
  val waveData = WaveReader.readWaveFromFile("./inp_16k.wav")
  val output = mutableListOf<Float>()

  val frameShift = denoiser.frameShiftInSamples
  var start = 0
  while (start < waveData.samples.size) {
    val end = minOf(start + frameShift, waveData.samples.size)
    val chunk = waveData.samples.copyOfRange(start, end)
    val denoised = denoiser.run(chunk, waveData.sampleRate)
    output.addAll(denoised.samples.asList())
    start = end
  }

  output.addAll(denoiser.flush().samples.asList())
  DenoisedAudio(output.toFloatArray(), denoiser.sampleRate).save(
    filename = "./enhanced-online-dpdfnet.wav"
  )
  println("saved to ./enhanced-online-dpdfnet.wav")

  denoiser.release()
}

fun createOnlineSpeechDenoiserGtcrn(): OnlineSpeechDenoiser {
  val config = OnlineSpeechDenoiserConfig(
      model = OfflineSpeechDenoiserModelConfig(
        gtcrn = OfflineSpeechDenoiserGtcrnModelConfig(
          model = "./gtcrn_simple.onnx"
        ),
        provider = "cpu",
        numThreads = 1,
      ),
  )

  return OnlineSpeechDenoiser(config = config)
}

fun createOnlineSpeechDenoiserDpdfNet(): OnlineSpeechDenoiser {
  val config = OnlineSpeechDenoiserConfig(
      model = OfflineSpeechDenoiserModelConfig(
        dpdfnet = OfflineSpeechDenoiserDpdfNetModelConfig(
          model = "./dpdfnet_baseline.onnx"
        ),
        provider = "cpu",
        numThreads = 1,
      ),
  )

  return OnlineSpeechDenoiser(config = config)
}
