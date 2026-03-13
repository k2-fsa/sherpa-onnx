package com.k2fsa.sherpa.onnx

fun main() {
  testAudioTagging()
}

fun testAudioTagging() {
  val config = AudioTaggingConfig(
      model=AudioTaggingModelConfig(
        zipformer=OfflineZipformerAudioTaggingModelConfig(
          model="./sherpa-onnx-zipformer-audio-tagging-2024-04-09/model.int8.onnx",
        ),
        numThreads=1,
        debug=true,
        provider="cpu",
      ),
      labels="./sherpa-onnx-zipformer-audio-tagging-2024-04-09/class_labels_indices.csv",
      topK=5,
   )
  val tagger = AudioTagging(config=config)

  val testFiles = arrayOf(
    "./sherpa-onnx-zipformer-audio-tagging-2024-04-09/test_wavs/1.wav",
    "./sherpa-onnx-zipformer-audio-tagging-2024-04-09/test_wavs/2.wav",
    "./sherpa-onnx-zipformer-audio-tagging-2024-04-09/test_wavs/3.wav",
    "./sherpa-onnx-zipformer-audio-tagging-2024-04-09/test_wavs/4.wav",
  )
  println("----------")
  for (waveFilename in testFiles) {
    val stream = tagger.createStream()

    val waveData = WaveReader.readWaveFromFile(
        filename = waveFilename,
    )

    stream.acceptWaveform(waveData.samples, sampleRate = waveData.sampleRate)
    val events = tagger.compute(stream)
    stream.release()

    println(waveFilename)
    for (event in events) {
      println("Name: ${event.name}, Index: ${event.index}, Probability: ${event.prob}")
    }

    println("----------")
  }

  tagger.release()
}

