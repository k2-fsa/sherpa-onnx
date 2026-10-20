//
//  ViewModel.swift
//  SherpaOnnxTtsPhonemize
//
//  Created by fangjun on 2023/11/23.
//

#if canImport(SherpaOnnx)
import SherpaOnnx
#elseif canImport(SherpaOnnxShared)
import SherpaOnnxShared
#else
#error("SherpaOnnx module not found. Please check your SPM dependency configuration.")
#endif
#if canImport(piper_phonemize)
import piper_phonemize
#else
#error("piper_phonemize module not found. Please check your SPM dependency configuration.")
#endif
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

/// How the input text is converted to phonemes before synthesis.
enum TextFrontend {
  /// Phonemize the text with piper-phonemize and pass the resulting phoneme
  /// codepoints to sherpa-onnx. No lexicon file is needed.
  case phonemize

  /// Let sherpa-onnx convert the text with the model's own lexicon (or its
  /// built-in frontend). piper-phonemize is not used.
  case lexicon
}

/// A TTS model together with the text frontend it expects.
struct TtsSetup {
  let tts: SherpaOnnxOfflineTtsWrapper
  let frontend: TextFrontend
}

/// Initialize the piper-phonemize phonemizer.
///
/// The espeak-ng-data used here is bundled inside the piper-phonemize package
/// and only feeds the piper phonemizer. sherpa-onnx itself does not use
/// espeak-ng anymore.
func initPiperPhonemize() {
  print("piper-phonemize version: \(piperPhonemizeGetVersionStr())")

  guard let dataDir = piperPhonemizeBundledDataDir() else {
    print("Error: bundled espeak-ng-data not found")
    return
  }

  let sampleRate = piperPhonemizeInitialize(dataDir: dataDir)
  if sampleRate < 0 {
    print("Error: failed to initialize piper-phonemize")
  }
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

  // This is a piper English model: the text is phonemized with piper-phonemize
  // and passed as phoneme codepoints, so no lexicon is used.
  let vits = sherpaOnnxOfflineTtsVitsModelConfig(model: model, tokens: tokens)
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

  let numFst = getResource("number", "fst")
  let dateFst = getResource("date", "fst")
  let phoneFst = getResource("phone", "fst")
  let ruleFsts = "\(dateFst),\(phoneFst),\(numFst)"

  let vits = sherpaOnnxOfflineTtsVitsModelConfig(
    model: model, lexicon: lexicon, tokens: tokens,
    noiseScale: 0.667,
    noiseScaleW: 0.8,
    lengthScale: 1.0
  )

  let modelConfig = sherpaOnnxOfflineTtsModelConfig(vits: vits)
  var config = sherpaOnnxOfflineTtsConfig(
    model: modelConfig,
    ruleFsts: ruleFsts
  )

  return SherpaOnnxOfflineTtsWrapper(config: &config)
}

func getTtsFor_matcha_icefall_zh_baker() -> SherpaOnnxOfflineTtsWrapper {
  // please see https://k2-fsa.github.io/sherpa/onnx/tts/pretrained_models/matcha.html#matcha-icefall-zh-baker-chinese-1-female-speaker

  let acousticModel = getResource("model-steps-3", "onnx")
  let vocoder = getResource("vocos-22khz-univ", "onnx")

  let tokens = getResource("tokens", "txt")
  let lexicon = getResource("lexicon", "txt")

  let numFst = getResource("number", "fst")
  let dateFst = getResource("date", "fst")
  let phoneFst = getResource("phone", "fst")
  let ruleFsts = "\(dateFst),\(phoneFst),\(numFst)"

  let matcha = sherpaOnnxOfflineTtsMatchaModelConfig(
    acousticModel: acousticModel,
    vocoder: vocoder,
    lexicon: lexicon,
    tokens: tokens
  )

  let modelConfig = sherpaOnnxOfflineTtsModelConfig(matcha: matcha)
  var config = sherpaOnnxOfflineTtsConfig(
    model: modelConfig,
    ruleFsts: ruleFsts
  )

  return SherpaOnnxOfflineTtsWrapper(config: &config)
}

func getTtsFor_kokoro_en_v0_19() -> SherpaOnnxOfflineTtsWrapper {
  // please see https://k2-fsa.github.io/sherpa/onnx/tts/pretrained_models/kokoro.html#kokoro-en-v0-19-english-11-speakers

  let model = getResource("model", "onnx")
  let voices = getResource("voices", "bin")

  // tokens.txt
  let tokens = getResource("tokens", "txt")

  // This is a piper-family English model: the text is phonemized with
  // piper-phonemize and passed as phoneme codepoints, so no lexicon is used.
  let kokoro = sherpaOnnxOfflineTtsKokoroModelConfig(
    model: model, voices: voices, tokens: tokens)
  let modelConfig = sherpaOnnxOfflineTtsModelConfig(kokoro: kokoro)
  var config = sherpaOnnxOfflineTtsConfig(model: modelConfig)

  return SherpaOnnxOfflineTtsWrapper(config: &config)
}

func getTtsFor_kokoro_multi_lang_v1_0() -> SherpaOnnxOfflineTtsWrapper {
  // please see https://k2-fsa.github.io/sherpa/onnx/tts/pretrained_models/kokoro.html

  let model = getResource("model", "onnx")
  let voices = getResource("voices", "bin")

  // tokens.txt
  let tokens = getResource("tokens", "txt")

  let lexicon_en = getResource("lexicon-us-en", "txt")
  let lexicon_zh = getResource("lexicon-zh", "txt")
  let lexicon = "\(lexicon_en),\(lexicon_zh)"


  let numFst = getResource("number-zh", "fst")
  let dateFst = getResource("date-zh", "fst")
  let phoneFst = getResource("phone-zh", "fst")
  let ruleFsts = "\(dateFst),\(phoneFst),\(numFst)"

  let kokoro = sherpaOnnxOfflineTtsKokoroModelConfig(
    model: model, voices: voices, tokens: tokens, lexicon: lexicon)
  let modelConfig = sherpaOnnxOfflineTtsModelConfig(kokoro: kokoro)
  var config = sherpaOnnxOfflineTtsConfig(model: modelConfig, ruleFsts: ruleFsts)

  return SherpaOnnxOfflineTtsWrapper(config: &config)
}

func getTtsForSupertonic() -> SherpaOnnxOfflineTtsWrapper {
  // please see https://k2-fsa.github.io/sherpa/onnx/tts/supertonic.html
  // Download: https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/sherpa-onnx-supertonic-3-tts-int8-2026-05-11.tar.bz2

  let durationPredictor = getResource("duration_predictor", "int8.onnx")
  let textEncoder = getResource("text_encoder", "int8.onnx")
  let vectorEstimator = getResource("vector_estimator", "int8.onnx")
  let vocoder = getResource("vocoder", "int8.onnx")
  let ttsJson = getResource("tts", "json")
  let unicodeIndexer = getResource("unicode_indexer", "bin")
  let voiceStyle = getResource("voice", "bin")

  let supertonic = sherpaOnnxOfflineTtsSupertonicModelConfig(
    durationPredictor: durationPredictor,
    textEncoder: textEncoder,
    vectorEstimator: vectorEstimator,
    vocoder: vocoder,
    ttsJson: ttsJson,
    unicodeIndexer: unicodeIndexer,
    voiceStyle: voiceStyle
  )
  let modelConfig = sherpaOnnxOfflineTtsModelConfig(numThreads: 2, supertonic: supertonic)
  var config = sherpaOnnxOfflineTtsConfig(model: modelConfig)
  return SherpaOnnxOfflineTtsWrapper(config: &config)
}

func createOfflineTts() -> TtsSetup {
  // Please enable only one of them.
  //
  // The text frontend is chosen per model: piper-phonemize emits espeak IPA,
  // which only the piper-family English models accept. Chinese/multi-lingual
  // models need their own lexicon; VCTK uses a coqui phone inventory; and
  // supertonic has its own text frontend.

  // return TtsSetup(tts: getTtsForSupertonic(), frontend: .lexicon)

  // return TtsSetup(tts: getTtsFor_kokoro_multi_lang_v1_0(), frontend: .lexicon)

  // return TtsSetup(tts: getTtsFor_kokoro_en_v0_19(), frontend: .phonemize)

  // return TtsSetup(tts: getTtsFor_matcha_icefall_zh_baker(), frontend: .lexicon)

  return TtsSetup(tts: getTtsFor_en_US_amy_low(), frontend: .phonemize)

  // return TtsSetup(tts: getTtsForVCTK(), frontend: .lexicon)

  // return TtsSetup(tts: getTtsForAishell3(), frontend: .lexicon)

  // return TtsSetup(tts: getTtsFor_zh_en_melo_tts(), frontend: .lexicon)

  // please add more models on need by following the above two examples
}
