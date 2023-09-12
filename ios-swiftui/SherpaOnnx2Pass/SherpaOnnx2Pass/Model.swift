import Foundation

func getResource(_ forResource: String, _ ofType: String) -> String {
  let path = Bundle.main.path(forResource: forResource, ofType: ofType)
  precondition(
    path != nil,
    "\(forResource).\(ofType) does not exist!\n" + "Remember to change \n"
      + "  Build Phases -> Copy Bundle Resources\n" + "to add it!"
  )
  return path!
}
/// Please refer to
/// https://k2-fsa.github.io/sherpa/onnx/pretrained_models/index.html
/// to download pre-trained models

/// sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20 (Bilingual, Chinese + English)
/// https://k2-fsa.github.io/sherpa/onnx/pretrained_models/zipformer-transducer-models.html
func getBilingualStreamingZhEnZipformer20230220() -> SherpaOnnxOnlineModelConfig {
  let encoder = getResource("encoder-epoch-99-avg-1.int8", "onnx")
  let decoder = getResource("decoder-epoch-99-avg-1", "onnx")
  let joiner = getResource("joiner-epoch-99-avg-1.int8", "onnx")
  let tokens = getResource("tokens", "txt")

  return sherpaOnnxOnlineModelConfig(
    tokens: tokens,
    transducer: sherpaOnnxOnlineTransducerModelConfig(
      encoder: encoder,
      decoder: decoder,
      joiner: joiner),
    numThreads: 1,
    modelType: "zipformer"
  )
}

/// csukuangfj/sherpa-onnx-streaming-zipformer-zh-14M-2023-02-23 (Chinese)
/// https://k2-fsa.github.io/sherpa/onnx/pretrained_models/online-transducer/zipformer-transducer-models.html#csukuangfj-sherpa-onnx-streaming-zipformer-zh-14m-2023-02-23-chinese

func getStreamingZh14MZipformer20230223() -> SherpaOnnxOnlineModelConfig {
  let encoder = getResource("encoder-epoch-99-avg-1.int8", "onnx")
  let decoder = getResource("decoder-epoch-99-avg-1", "onnx")
  let joiner = getResource("joiner-epoch-99-avg-1.int8", "onnx")
  let tokens = getResource("tokens", "txt")

  return sherpaOnnxOnlineModelConfig(
    tokens: tokens,
    transducer: sherpaOnnxOnlineTransducerModelConfig(
      encoder: encoder,
      decoder: decoder,
      joiner: joiner),
    numThreads: 1,
    modelType: "zipformer"
  )
}

/// csukuangfj/sherpa-onnx-streaming-zipformer-en-20M-2023-02-17 (English)
/// https://k2-fsa.github.io/sherpa/onnx/pretrained_models/online-transducer/zipformer-transducer-models.html#csukuangfj-sherpa-onnx-streaming-zipformer-en-20m-2023-02-17-english

func getStreamingEn20MZipformer20230217() -> SherpaOnnxOnlineModelConfig {
  let encoder = getResource("encoder-epoch-99-avg-1.int8", "onnx")
  let decoder = getResource("decoder-epoch-99-avg-1", "onnx")
  let joiner = getResource("joiner-epoch-99-avg-1", "onnx")
  let tokens = getResource("tokens", "txt")

  return sherpaOnnxOnlineModelConfig(
    tokens: tokens,
    transducer: sherpaOnnxOnlineTransducerModelConfig(
      encoder: encoder,
      decoder: decoder,
      joiner: joiner),
    numThreads: 1,
    modelType: "zipformer"
  )
}

/// ========================================
///   Non-streaming models
/// ========================================

/// csukuangfj/sherpa-onnx-paraformer-zh-2023-03-28 (Chinese)
/// https://k2-fsa.github.io/sherpa/onnx/pretrained_models/offline-paraformer/paraformer-models.html#csukuangfj-sherpa-onnx-paraformer-zh-2023-03-28-chinese
func getNonStreamingZhParaformer20230328() -> SherpaOnnxOfflineModelConfig {
  let model = getResource("model.int8", "onnx")
  let tokens = getResource("paraformer-tokens", "txt")

  return sherpaOnnxOfflineModelConfig(
    tokens: tokens,
    paraformer: sherpaOnnxOfflineParaformerModelConfig(
      model: model),
    numThreads: 1,
    modelType: "paraformer"
  )
}

// https://k2-fsa.github.io/sherpa/onnx/pretrained_models/whisper/tiny.en.html#tiny-en
// English, int8 encoder and decoder
func getNonStreamingWhisperTinyEn() -> SherpaOnnxOfflineModelConfig {
  let encoder = getResource("tiny.en-encoder.int8", "onnx")
  let decoder = getResource("tiny.en-decoder.int8", "onnx")
  let tokens = getResource("tiny.en-tokens", "txt")

  return sherpaOnnxOfflineModelConfig(
    tokens: tokens,
    whisper: sherpaOnnxOfflineWhisperModelConfig(
      encoder: encoder,
      decoder: decoder
    ),
    numThreads: 1,
    modelType: "whisper"
  )
}

// icefall-asr-multidataset-pruned_transducer_stateless7-2023-05-04 (English)
// https://k2-fsa.github.io/sherpa/onnx/pretrained_models/offline-transducer/zipformer-transducer-models.html#icefall-asr-multidataset-pruned-transducer-stateless7-2023-05-04-english

func getNonStreamingEnZipformer20230504() -> SherpaOnnxOfflineModelConfig {
  let encoder = getResource("encoder-epoch-30-avg-4.int8", "onnx")
  let decoder = getResource("decoder-epoch-30-avg-4", "onnx")
  let joiner = getResource("joiner-epoch-30-avg-4", "onnx")
  let tokens = getResource("non-streaming-zipformer-tokens", "txt")

  return sherpaOnnxOfflineModelConfig(
    tokens: tokens,
    transducer: sherpaOnnxOfflineTransducerModelConfig(
      encoder: encoder,
      decoder: decoder,
      joiner: joiner),
    numThreads: 1,
    modelType: "zipformer"
  )
}

/// Please refer to
/// https://k2-fsa.github.io/sherpa/onnx/pretrained_models/index.html
/// to add more models if you need
