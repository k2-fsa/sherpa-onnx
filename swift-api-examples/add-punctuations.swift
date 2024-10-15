func run() {
  let model = "./sherpa-onnx-punct-ct-transformer-zh-en-vocab272727-2024-04-12/model.onnx"
  let modelConfig = sherpaOnnxOfflinePunctuationModelConfig(
    ctTransformer: model,
    numThreads: 1,
    debug: 1,
    provider: "cpu"
  )
  var config = sherpaOnnxOfflinePunctuationConfig(model: modelConfig)

  let punct = SherpaOnnxOfflinePunctuationWrapper(config: &config)

  let textList = [
    "这是一个测试你好吗How are you我很好thank you are you ok谢谢你",
    "我们都是木头人不会说话不会动",
    "The African blogosphere is rapidly expanding bringing more voices online in the form of commentaries opinions analyses rants and poetry",
  ]

  for i in 0..<textList.count {
    let t = punct.addPunct(text: textList[i])
    print("\nresult is:\n\(t)")
  }

}

@main
struct App {
  static func main() {
    run()
  }
}
