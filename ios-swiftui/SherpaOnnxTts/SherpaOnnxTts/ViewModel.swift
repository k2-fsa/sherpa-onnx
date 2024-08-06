//
//  ViewModel.swift
//  SherpaOnnxTts
//
//  Created by fangjun on 2023/11/23.
//

import Foundation

// used to get the path to espeak-ng-data
func resourceURL(to path: String) -> String {
  return URL(string: path, relativeTo: Bundle.main.resourceURL)!.path
}

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

  let model = getResource("model", "onnx")

  // lexicon.txt
  let lexicon = getResource("lexicon", "txt")

  // tokens.txt
  let tokens = getResource("tokens", "txt")

  // rule.fst
  let ruleFsts = getResource("rule", "fst")

  // rule.far
  let ruleFars = getResource("rule", "far")

  let vits = sherpaOnnxOfflineTtsVitsModelConfig(model: model, lexicon: lexicon, tokens: tokens)
  let modelConfig = sherpaOnnxOfflineTtsModelConfig(vits: vits)
  var config = sherpaOnnxOfflineTtsConfig(
    model: modelConfig,
    ruleFsts: ruleFsts,
    ruleFars: ruleFars
  )
  return SherpaOnnxOfflineTtsWrapper(config: &config)
}

// https://github.com/k2-fsa/sherpa-onnx/releases/tag/tts-models
func getTtsFor_en_US_amy_low() -> SherpaOnnxOfflineTtsWrapper {
  // please see  https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/vits-piper-en_US-amy-low.tar.bz2

  let model = getResource("en_US-amy-low", "onnx")

  // tokens.txt
  let tokens = getResource("tokens", "txt")

  // in this case, we don't need lexicon.txt
  let dataDir = resourceURL(to: "espeak-ng-data")

  let vits = sherpaOnnxOfflineTtsVitsModelConfig(
    model: model, lexicon: "", tokens: tokens, dataDir: dataDir)
  let modelConfig = sherpaOnnxOfflineTtsModelConfig(vits: vits)
  var config = sherpaOnnxOfflineTtsConfig(model: modelConfig)

  return SherpaOnnxOfflineTtsWrapper(config: &config)
}

// https://k2-fsa.github.io/sherpa/onnx/tts/pretrained_models/vits.html#vits-melo-tts-zh-en-chinese-english-1-speaker
func getTtsFor_zh_en_melo_tts() -> SherpaOnnxOfflineTtsWrapper {
  // please see https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/vits-melo-tts-zh_en.tar.bz2

  let model = getResource("model", "onnx")

  let tokens = getResource("tokens", "txt")
  let lexicon = getResource("lexicon", "txt")

  let dictDir = resourceURL(to: "dict")

  let numFst = getResource("number", "fst")
  let dateFst = getResource("date", "fst")
  let phoneFst = getResource("phone", "fst")
  let ruleFsts = "\(dateFst),\(phoneFst),\(numFst)"

  let vits = sherpaOnnxOfflineTtsVitsModelConfig(
    model: model, lexicon: lexicon, tokens: tokens,
    dataDir: "",
    noiseScale: 0.667,
    noiseScaleW: 0.8,
    lengthScale: 1.0,
    dictDir: dictDir
  )

  let modelConfig = sherpaOnnxOfflineTtsModelConfig(vits: vits)
  var config = sherpaOnnxOfflineTtsConfig(
    model: modelConfig,
    ruleFsts: ruleFsts
  )

  return SherpaOnnxOfflineTtsWrapper(config: &config)
}

func createOfflineTts() -> SherpaOnnxOfflineTtsWrapper {
  // Please enable only one of them

  return getTtsFor_en_US_amy_low()

  // return getTtsForVCTK()

  // return getTtsForAishell3()

  // return getTtsFor_zh_en_melo_tts()

  // please add more models on need by following the above two examples
}
