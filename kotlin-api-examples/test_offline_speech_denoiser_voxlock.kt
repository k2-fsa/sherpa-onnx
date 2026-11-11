package com.k2fsa.sherpa.onnx

// VoxLock is a target-speaker voice isolation model. It keeps the voice of one
// speaker, e.g., the caller in a phone call, and removes the other voices.
//
// The model is from https://github.com/arbazsiddiqui/voxlock
//
// This example does not provide a voice sample, so VoxLock takes the loudest
// 3 seconds in the first 15 seconds of the input as the speaker to keep. See
// test_offline_speech_denoiser_voxlock_with_voice_sample.kt if you have a
// voice sample of the speaker.
//
// Please download test files in this script from
// https://github.com/k2-fsa/sherpa-onnx/releases/tag/speech-enhancement-models

fun main() {
  test()
}

fun test() {
  val denoiser = createOfflineSpeechDenoiser()

  val waveFilename = "./sherpa-onnx-voxlock-int8/test_wavs/en-combined.wav"
  val waveData = WaveReader.readWaveFromFile(filename = waveFilename)

  val begin = System.currentTimeMillis()
  val denoised = denoiser.run(waveData.samples, waveData.sampleRate)
  val elapsedMs = System.currentTimeMillis() - begin

  denoised.save(filename = "./voxlock-enhanced-16k.wav")

  val elapsedSeconds = elapsedMs / 1000.0f
  val duration = waveData.samples.size / waveData.sampleRate.toFloat()
  println("Saved to ./voxlock-enhanced-16k.wav")
  println("Elapsed seconds: " + "%.3f".format(elapsedSeconds))
  println("Audio duration in seconds: " + "%.3f".format(duration))
  println(
    "RTF: " + "%.3f".format(elapsedSeconds) + "/" + "%.3f".format(duration) + " = " +
      "%.3f".format(elapsedSeconds / duration)
  )

  denoiser.release()
}

fun createOfflineSpeechDenoiser(): OfflineSpeechDenoiser {
  val config = OfflineSpeechDenoiserConfig(
    model = OfflineSpeechDenoiserModelConfig(
      voxlock = OfflineSpeechDenoiserVoxLockModelConfig(
        enrollModel = "./sherpa-onnx-voxlock-int8/enroll.onnx",
        chunkModel = "./sherpa-onnx-voxlock-int8/chunk_int8.onnx",
      ),
      provider = "cpu",
      numThreads = 1,
    ),
  )

  println(config)

  return OfflineSpeechDenoiser(config = config)
}
