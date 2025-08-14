/// swift-api-examples/compute-speaker-embeddings.swift
/// Copyright (c)  2025  Xiaomi Corporation
/*
Please download test files used in this script from

https://github.com/k2-fsa/sherpa-onnx/releases/tag/speaker-recongition-models
*/
func cosineSimilarity(_ a: [Float], _ b: [Float]) -> Float {
  precondition(a.count == b.count, "Vectors must have the same length")

  // Dot product
  let dotProduct = zip(a, b).reduce(0) { $0 + $1.0 * $1.1 }

  // Magnitudes
  let magA = sqrt(a.reduce(0) { $0 + $1 * $1 })
  let magB = sqrt(b.reduce(0) { $0 + $1 * $1 })

  // Avoid division by zero
  guard magA > 0 && magB > 0 else { return 0 }

  return dotProduct / (magA * magB)
}

func computeEmbedding(extractor: SherpaOnnxSpeakerEmbeddingExtractorWrapper, waveFilename: String)
  -> [Float]
{
  let audio = SherpaOnnxWaveWrapper.readWave(filename: waveFilename)
  let stream = extractor.createStream()
  stream.acceptWaveform(samples: audio.samples, sampleRate: audio.sampleRate)
  stream.inputFinished()
  return extractor.compute(stream: stream)
}

func run() {
  let model = "./wespeaker_zh_cnceleb_resnet34.onnx"
  var config = sherpaOnnxSpeakerEmbeddingExtractorConfig(model: model)
  let extractor = SherpaOnnxSpeakerEmbeddingExtractorWrapper(config: &config)
  let embedding1 = computeEmbedding(extractor: extractor, waveFilename: "./fangjun-sr-1.wav")
  let embedding2 = computeEmbedding(extractor: extractor, waveFilename: "./fangjun-sr-2.wav")
  let embedding3 = computeEmbedding(extractor: extractor, waveFilename: "./leijun-sr-1.wav")

  let score12 = cosineSimilarity(embedding1, embedding2)
  let score13 = cosineSimilarity(embedding1, embedding3)
  let score23 = cosineSimilarity(embedding2, embedding3)

  print("Score between spk1 and spk2: \(score12)")
  print("Score between spk1 and spk3: \(score13)")
  print("Score between spk2 and spk3: \(score23)")
}

@main
struct App {
  static func main() {
    run()
  }
}
