package com.k2fsa.sherpa.onnx

// This file shows how to use a non-streaming Zipformer CTC model with an HLG
// decoding graph via sherpa-onnx's Kotlin API.
//
// The HLG graph is passed to the C API through ctcFstDecoderConfig.

fun main() {
  val recognizer = createOfflineRecognizer()
  val waveFilename = "./sherpa-onnx-zipformer-ctc-en-2023-10-02/test_wavs/0.wav"

  val waveData = WaveReader.readWaveFromFile(
      filename = waveFilename,
  )

  var stream = recognizer.createStream()
  stream.acceptWaveform(waveData.samples, sampleRate = waveData.sampleRate)
  recognizer.decode(stream)

  var result = recognizer.getResult(stream)
  println(result)

  stream.release()
  recognizer.release()
}

fun createOfflineRecognizer(): OfflineRecognizer {
  val config = OfflineRecognizerConfig(
      modelConfig = OfflineModelConfig(
          zipformerCtc = OfflineZipformerCtcModelConfig(
              model = "./sherpa-onnx-zipformer-ctc-en-2023-10-02/model.int8.onnx",
          ),
          tokens = "./sherpa-onnx-zipformer-ctc-en-2023-10-02/tokens.txt",
          numThreads = 1,
          debug = true,
      ),
      // This is what the example exercises: the graph and maxActive reach the
      // C API through ctcFstDecoderConfig.
      ctcFstDecoderConfig = OfflineCtcFstDecoderConfig(
          graph = "./sherpa-onnx-zipformer-ctc-en-2023-10-02/HLG.fst",
          maxActive = 3000,
      ),
  )

  return OfflineRecognizer(config = config)
}
