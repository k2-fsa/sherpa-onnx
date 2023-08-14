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
func getBilingualStreamZhEnZipformer20230220() -> SherpaOnnxOnlineModelConfig {
  let encoder = getResource("encoder-epoch-99-avg-1", "onnx")
  let decoder = getResource("decoder-epoch-99-avg-1", "onnx")
  let joiner = getResource("joiner-epoch-99-avg-1", "onnx")
  let tokens = getResource("tokens", "txt")

  return sherpaOnnxOnlineModelConfig(
    tokens: tokens,
    transducer: sherpaOnnxOnlineTransducerModelConfig(
      encoder: encoder,
      decoder: decoder,
      joiner: joiner
    ),
    numThreads: 1,
    modelType: "zipformer"
  )
}

func getZhZipformer20230615() -> SherpaOnnxOnlineModelConfig {
  let encoder = getResource("encoder-epoch-12-avg-4-chunk-16-left-128", "onnx")
  let decoder = getResource("decoder-epoch-12-avg-4-chunk-16-left-128", "onnx")
  let joiner = getResource("joiner-epoch-12-avg-4-chunk-16-left-128", "onnx")
  let tokens = getResource("tokens", "txt")

  return sherpaOnnxOnlineModelConfig(
    tokens: tokens,
    transducer: sherpaOnnxOnlineTransducerModelConfig(
      encoder: encoder,
      decoder: decoder,
      joiner: joiner
    ),
    numThreads: 1,
    modelType: "zipformer2"
  )
}

func getZhZipformer20230615Int8() -> SherpaOnnxOnlineModelConfig {
  let encoder = getResource("encoder-epoch-12-avg-4-chunk-16-left-128.int8", "onnx")
  let decoder = getResource("decoder-epoch-12-avg-4-chunk-16-left-128", "onnx")
  let joiner = getResource("joiner-epoch-12-avg-4-chunk-16-left-128", "onnx")
  let tokens = getResource("tokens", "txt")

  return sherpaOnnxOnlineModelConfig(
    tokens: tokens,
    transducer: sherpaOnnxOnlineTransducerModelConfig(
      encoder: encoder,
      decoder: decoder,
      joiner: joiner),
    numThreads: 1,
    modelType: "zipformer2"
  )
}

func getEnZipformer20230626() -> SherpaOnnxOnlineModelConfig {
  let encoder = getResource("encoder-epoch-99-avg-1-chunk-16-left-128", "onnx")
  let decoder = getResource("decoder-epoch-99-avg-1-chunk-16-left-128", "onnx")
  let joiner = getResource("joiner-epoch-99-avg-1-chunk-16-left-128", "onnx")
  let tokens = getResource("tokens", "txt")

  return sherpaOnnxOnlineModelConfig(
    tokens: tokens,
    transducer: sherpaOnnxOnlineTransducerModelConfig(
      encoder: encoder,
      decoder: decoder,
      joiner: joiner),
    numThreads: 1,
    modelType: "zipformer2"
  )
}

func getBilingualStreamingZhEnParaformer() -> SherpaOnnxOnlineModelConfig {
  let encoder = getResource("encoder.int8", "onnx")
  let decoder = getResource("decoder.int8", "onnx")
  let tokens = getResource("tokens", "txt")

  return sherpaOnnxOnlineModelConfig(
    tokens: tokens,
    paraformer: sherpaOnnxOnlineParaformerModelConfig(
      encoder: encoder,
      decoder: decoder),
    numThreads: 1,
    modelType: "paraformer"
  )
}

/// Please refer to
/// https://k2-fsa.github.io/sherpa/onnx/pretrained_models/index.html
/// to add more models if you need
