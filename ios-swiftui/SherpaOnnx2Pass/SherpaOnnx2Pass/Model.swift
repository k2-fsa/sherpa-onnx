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


/// Please refer to
/// https://k2-fsa.github.io/sherpa/onnx/pretrained_models/index.html
/// to add more models if you need
