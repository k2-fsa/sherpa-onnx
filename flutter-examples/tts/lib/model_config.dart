// Copyright (c)  2026  Xiaomi Corporation
// Shared model selection — edit this file to change the TTS model.
// Both native (model.dart) and web (model_web.dart) use this.
//
import 'package:sherpa_onnx/sherpa_onnx.dart';

// Change the index below to select a different model.
/// Select which TTS model to use (0-10).
const int selectedModelIndex = 0;

/// Model directory name, extracted from the first non-empty model path.
final String selectedModelDir = () {
  final m = selectedTtsConfig.model;
  final path = m.vits.model.isNotEmpty
      ? m.vits.model
      : m.matcha.acousticModel.isNotEmpty
          ? m.matcha.acousticModel
          : m.kokoro.model.isNotEmpty
              ? m.kokoro.model
              : m.kitten.model.isNotEmpty
                  ? m.kitten.model
                  : m.pocket.lmFlow.isNotEmpty
                      ? m.pocket.lmFlow
                      : m.supertonic.durationPredictor.isNotEmpty
                          ? m.supertonic.durationPredictor
                          : m.zipvoice.encoder.isNotEmpty
                              ? m.zipvoice.encoder
                              : '';
  return path.contains('/') ? path.split('/').first : path;
}();

/// Download URL for the selected model.
final String selectedModelUrl =
    'https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/$selectedModelDir.tar.bz2';

/// Available TTS models.
final OfflineTtsConfig selectedTtsConfig = switch (selectedModelIndex) {
  // ── VITS Piper (English) ──────────────────────────────────────────────
  0 => OfflineTtsConfig(
    model: OfflineTtsModelConfig(
      vits: OfflineTtsVitsModelConfig(
        model: 'vits-piper-en_US-amy-low/en_US-amy-low.onnx',
        tokens: 'vits-piper-en_US-amy-low/tokens.txt',
        dataDir: 'vits-piper-en_US-amy-low/espeak-ng-data',
      ),
      numThreads: 2,
      debug: true,
    ),
    maxNumSenetences: 1,
  ),

  // ── VITS Piper (Chinese) ─────────────────────────────────────────────
  1 => OfflineTtsConfig(
    model: OfflineTtsModelConfig(
      vits: OfflineTtsVitsModelConfig(
        model: 'vits-piper-zh_CN-xiao_ya-medium/zh_CN-xiao_ya-medium.onnx',
        tokens: 'vits-piper-zh_CN-xiao_ya-medium/tokens.txt',
        lexicon: 'vits-piper-zh_CN-xiao_ya-medium/lexicon.txt',
      ),
      numThreads: 2,
      debug: true,
    ),
    ruleFsts: 'vits-piper-zh_CN-xiao_ya-medium/phone.fst,vits-piper-zh_CN-xiao_ya-medium/date.fst,vits-piper-zh_CN-xiao_ya-medium/number.fst',
    maxNumSenetences: 1,
  ),

  // ── VITS Piper (English, libritts) ────────────────────────────────────
  2 => OfflineTtsConfig(
    model: OfflineTtsModelConfig(
      vits: OfflineTtsVitsModelConfig(
        model: 'vits-piper-en_US-libritts_r-medium/en_US-libritts_r-medium.onnx',
        tokens: 'vits-piper-en_US-libritts_r-medium/tokens.txt',
        dataDir: 'vits-piper-en_US-libritts_r-medium/espeak-ng-data',
      ),
      numThreads: 2,
      debug: true,
    ),
    maxNumSenetences: 1,
  ),

  // ── VITS (English, inflect-nano-v2) ──────────────────────────────────
  // https://k2-fsa.github.io/sherpa/onnx/tts/all/English/vits-inflect-en-nano-v2.html
  3 => OfflineTtsConfig(
    model: OfflineTtsModelConfig(
      vits: OfflineTtsVitsModelConfig(
        model: 'vits-inflect-en-nano-v2/model.onnx',
        tokens: 'vits-inflect-en-nano-v2/tokens.txt',
        dataDir: 'vits-inflect-en-nano-v2/espeak-ng-data',
      ),
      numThreads: 2,
      debug: true,
    ),
    maxNumSenetences: 1,
  ),

  // ── Kokoro (English) ──────────────────────────────────────────────────
  // warning: It is super slow with single threaded wasm
  4 => OfflineTtsConfig(
    model: OfflineTtsModelConfig(
      kokoro: OfflineTtsKokoroModelConfig(
        model: 'kokoro-int8-en-v0_19/model.int8.onnx',
        voices: 'kokoro-int8-en-v0_19/voices.bin',
        tokens: 'kokoro-int8-en-v0_19/tokens.txt',
        dataDir: 'kokoro-int8-en-v0_19/espeak-ng-data',
      ),
      numThreads: 2,
      debug: true,
    ),
    maxNumSenetences: 1,
  ),

  // ── Kokoro (Chinese + English) ────────────────────────────────────────
  // warning: It is super slow with single threaded wasm
  5 => OfflineTtsConfig(
    model: OfflineTtsModelConfig(
      kokoro: OfflineTtsKokoroModelConfig(
        model: 'kokoro-multi-lang-v1_0/model.onnx',
        voices: 'kokoro-multi-lang-v1_0/voices.bin',
        tokens: 'kokoro-multi-lang-v1_0/tokens.txt',
        dataDir: 'kokoro-multi-lang-v1_0/espeak-ng-data',
        lexicon: 'kokoro-multi-lang-v1_0/lexicon-us-en.txt,kokoro-multi-lang-v1_0/lexicon-zh.txt',
      ),
      numThreads: 2,
      debug: true,
    ),
    maxNumSenetences: 1,
  ),

  // ── MatchaTTS (English) ───────────────────────────────────────────────
  6 => OfflineTtsConfig(
    model: OfflineTtsModelConfig(
      matcha: OfflineTtsMatchaModelConfig(
        acousticModel: 'matcha-icefall-en_US-ljspeech/model-steps-3.onnx',
        vocoder: 'vocos-22khz-univ.onnx',
        tokens: 'matcha-icefall-en_US-ljspeech/tokens.txt',
        dataDir: 'matcha-icefall-en_US-ljspeech/espeak-ng-data',
      ),
      numThreads: 2,
      debug: true,
    ),
    maxNumSenetences: 1,
  ),

  // ── MatchaTTS (Chinese + English) ────────────────────────────────────
  // https://k2-fsa.github.io/sherpa/onnx/tts/all/Chinese-English/matcha-icefall-zh-en.html
  7 => OfflineTtsConfig(
    model: OfflineTtsModelConfig(
      matcha: OfflineTtsMatchaModelConfig(
        acousticModel: 'matcha-icefall-zh-en/model-steps-3.onnx',
        vocoder: 'vocos-16khz-univ.onnx',
        lexicon: 'matcha-icefall-zh-en/lexicon.txt',
        tokens: 'matcha-icefall-zh-en/tokens.txt',
        dataDir: 'matcha-icefall-zh-en/espeak-ng-data',
      ),
      numThreads: 2,
      debug: true,
    ),
    ruleFsts: 'matcha-icefall-zh-en/phone-zh.fst,matcha-icefall-zh-en/date-zh.fst,matcha-icefall-zh-en/number-zh.fst',
    maxNumSenetences: 1,
  ),

  // ── KittenTTS (English) ───────────────────────────────────────────────
  8 => OfflineTtsConfig(
    model: OfflineTtsModelConfig(
      kitten: OfflineTtsKittenModelConfig(
        model: 'kitten-nano-en-v0_1-fp16/model.fp16.onnx',
        voices: 'kitten-nano-en-v0_1-fp16/voices.bin',
        tokens: 'kitten-nano-en-v0_1-fp16/tokens.txt',
        dataDir: 'kitten-nano-en-v0_1-fp16/espeak-ng-data',
      ),
      numThreads: 2,
      debug: true,
    ),
    maxNumSenetences: 1,
  ),

  // ── Pocket TTS (English) ──────────────────────────────────────────────
  9 => OfflineTtsConfig(
    model: OfflineTtsModelConfig(
      pocket: OfflineTtsPocketModelConfig(
        lmFlow: 'sherpa-onnx-pocket-tts-int8-2026-01-26/lm_flow.int8.onnx',
        lmMain: 'sherpa-onnx-pocket-tts-int8-2026-01-26/lm_main.int8.onnx',
        encoder: 'sherpa-onnx-pocket-tts-int8-2026-01-26/encoder.onnx',
        decoder: 'sherpa-onnx-pocket-tts-int8-2026-01-26/decoder.int8.onnx',
        textConditioner: 'sherpa-onnx-pocket-tts-int8-2026-01-26/text_conditioner.onnx',
        vocabJson: 'sherpa-onnx-pocket-tts-int8-2026-01-26/vocab.json',
        tokenScoresJson: 'sherpa-onnx-pocket-tts-int8-2026-01-26/token_scores.json',
        voiceEmbeddingCacheCapacity: 50,
      ),
      numThreads: 2,
      debug: true,
    ),
  ),

  // ── Supertonic TTS (English) ──────────────────────────────────────────
  10 => OfflineTtsConfig(
    model: OfflineTtsModelConfig(
      supertonic: OfflineTtsSupertonicModelConfig(
        durationPredictor: 'sherpa-onnx-supertonic-3-tts-int8-2026-05-11/duration_predictor.int8.onnx',
        textEncoder: 'sherpa-onnx-supertonic-3-tts-int8-2026-05-11/text_encoder.int8.onnx',
        vectorEstimator: 'sherpa-onnx-supertonic-3-tts-int8-2026-05-11/vector_estimator.int8.onnx',
        vocoder: 'sherpa-onnx-supertonic-3-tts-int8-2026-05-11/vocoder.int8.onnx',
        ttsJson: 'sherpa-onnx-supertonic-3-tts-int8-2026-05-11/tts.json',
        unicodeIndexer: 'sherpa-onnx-supertonic-3-tts-int8-2026-05-11/unicode_indexer.bin',
        voiceStyle: 'sherpa-onnx-supertonic-3-tts-int8-2026-05-11/voice.bin',
      ),
      numThreads: 2,
      debug: true,
    ),
  ),

  _ => throw ArgumentError('Invalid selectedModelIndex: $selectedModelIndex. Must be 0-10.'),
};
