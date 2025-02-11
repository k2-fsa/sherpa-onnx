func run() {
    let model = "./sherpa-onnx-online-punct-en-2024-08-06/model.onnx"
    let bpe = "./sherpa-onnx-online-punct-en-2024-08-06/bpe.vocab"
    
    // Create model config
    let modelConfig = sherpaOnnxOnlinePunctuationModelConfig(
        cnnBiLstm: model,
        bpeVocab: bpe
    )
    
    // Create punctuation config
    var config = sherpaOnnxOnlinePunctuationConfig(model: modelConfig)
    
    // Create punctuation instance
    let punct = SherpaOnnxOnlinePunctuationWrapper(config: &config)
    
    // Test texts
    let textList = [
        "how are you i am fine thank you",
        "The African blogosphere is rapidly expanding bringing more voices online in the form of commentaries opinions analyses rants and poetry"
    ]
    
    // Process each text
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
