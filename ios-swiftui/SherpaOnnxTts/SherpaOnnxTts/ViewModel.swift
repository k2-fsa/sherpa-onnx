//
//  ViewModel.swift
//  SherpaOnnxTts
//
//  Created by fangjun on 2023/11/23.
//

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
/// https://k2-fsa.github.io/sherpa/onnx/tts/pretrained_models/index.html
/// to download pre-trained models

func getTtsForVCTK() -> SherpaOnnxOfflineTtsWrapper {
  // See the following link
  // https://k2-fsa.github.io/sherpa/onnx/tts/pretrained_models/vits.html#vctk-english-multi-speaker-109-speakers

  // vits-vctk.onnx
  let model = getResource("vits-vctk", "onnx")

  // lexicon.txt
  let lexicon = getResource("lexicon", "txt")

  // tokens.txt
  let tokens = getResource("tokens", "txt")

  let vits = sherpaOnnxOfflineTtsVitsModelConfig(model: model, lexicon: lexicon, tokens: tokens)
  let modelConfig = sherpaOnnxOfflineTtsModelConfig(vits: vits)
  var config = sherpaOnnxOfflineTtsConfig(model: modelConfig)
  return SherpaOnnxOfflineTtsWrapper(config: &config)
}

func getTtsForAishell3() -> SherpaOnnxOfflineTtsWrapper {
  // See the following link
  // https://k2-fsa.github.io/sherpa/onnx/tts/pretrained_models/vits.html#vits-model-aishell3

  // vits-vctk.onnx
  let model = getResource("vits-aishell3", "onnx")

  // lexicon.txt
  let lexicon = getResource("lexicon", "txt")

  // tokens.txt
  let tokens = getResource("tokens", "txt")

  let vits = sherpaOnnxOfflineTtsVitsModelConfig(model: model, lexicon: lexicon, tokens: tokens)
  let modelConfig = sherpaOnnxOfflineTtsModelConfig(vits: vits)
  var config = sherpaOnnxOfflineTtsConfig(model: modelConfig)
  return SherpaOnnxOfflineTtsWrapper(config: &config)
}

func createOfflineTts() -> SherpaOnnxOfflineTtsWrapper {
  return getTtsForVCTK()

  // return getTtsForAishell3()

  // please add more models on need by following the above two examples
}
