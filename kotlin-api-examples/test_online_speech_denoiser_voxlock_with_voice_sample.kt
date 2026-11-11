package com.k2fsa.sherpa.onnx

// VoxLock is a target-speaker voice isolation model. It keeps the voice of one
// speaker, e.g., the caller in a phone call, and removes the other voices.
//
// The model is from https://github.com/arbazsiddiqui/voxlock
//
// This example provides a voice sample of the speaker to keep, so VoxLock
// isolates that speaker from the first sample. The voice sample must have at
// least 3 seconds of the speaker's voice; from a longer file, VoxLock uses the
// loudest 3 seconds within its first 30 seconds. See
// test_online_speech_denoiser_voxlock.kt if you don't have a voice sample.
//
// Please download test files in this script from
// https://github.com/k2-fsa/sherpa-onnx/releases/tag/speech-enhancement-models

fun main() {
  test()
}

fun test() {
  val denoiser = createOnlineSpeechDenoiser()
  val waveData = WaveReader.readWaveFromFile("./sherpa-onnx-voxlock-int8/test_wavs/en-combined.wav")
  val output = mutableListOf<Float>()
  val frameShift = denoiser.frameShiftInSamples

  val begin = System.currentTimeMillis()
  var start = 0
  while (start < waveData.samples.size) {
    val end = minOf(start + frameShift, waveData.samples.size)
    val chunk = waveData.samples.copyOfRange(start, end)
    val denoised = denoiser.run(chunk, waveData.sampleRate)
    output.addAll(denoised.samples.asList())
    start = end
  }

  output.addAll(denoiser.flush().samples.asList())
  val elapsedMs = System.currentTimeMillis() - begin

  DenoisedAudio(output.toFloatArray(), denoiser.sampleRate).save(
    filename = "./voxlock-enhanced-online-with-voice-sample-16k.wav"
  )

  val elapsedSeconds = elapsedMs / 1000.0f
  val duration = waveData.samples.size / waveData.sampleRate.toFloat()
  println("Saved to ./voxlock-enhanced-online-with-voice-sample-16k.wav")
  println("Elapsed seconds: " + "%.3f".format(elapsedSeconds))
  println("Audio duration in seconds: " + "%.3f".format(duration))
  println(
    "RTF: " + "%.3f".format(elapsedSeconds) + "/" + "%.3f".format(duration) + " = " +
      "%.3f".format(elapsedSeconds / duration)
  )

  denoiser.release()
}

fun createOnlineSpeechDenoiser(): OnlineSpeechDenoiser {
  val config = OnlineSpeechDenoiserConfig(
    model = OfflineSpeechDenoiserModelConfig(
      voxlock = OfflineSpeechDenoiserVoxLockModelConfig(
        enrollModel = "./sherpa-onnx-voxlock-int8/enroll.onnx",
        chunkModel = "./sherpa-onnx-voxlock-int8/chunk_int8.onnx",
        voiceSample = "./sherpa-onnx-voxlock-int8/test_wavs/en-voice-sample-f.wav",
      ),
      provider = "cpu",
      numThreads = 1,
    ),
  )

  return OnlineSpeechDenoiser(config = config)
}
