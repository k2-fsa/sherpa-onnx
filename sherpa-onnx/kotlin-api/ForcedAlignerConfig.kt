package com.k2fsa.sherpa.onnx

/**
 * Configuration for [ForcedAligner]: a forced-alignment acoustic model
 * directory, a pronunciation dictionary and the default alignment options.
 *
 * @property modelDir directory of one forced-alignment acoustic model.
 *
 * Required files:
 *  - `meta.json`: feature chain + silence/OOV parameters (parsed once at
 *    construction, field by field)
 *  - `final.alimdl`: alignment model, kaldi format (TransitionModel +
 *    AmDiagGmm). Note: NOT `final.mdl`
 *  - `tree`: context dependency tree (ContextDependency)
 *  - `lda.mat`: LDA transform of the feature chain; required iff the
 *    feature chain ends in splice+LDA
 *
 * Optional file:
 *  - `phones.txt`: phone table (symbol to id). When missing, the phone ids
 *    are taken from the `phones` list of `meta.json` following the kaldi
 *    convention of these models (`<eps>=0`, `sil=1`, `spn=2`, then the
 *    lexical phones from 3). Models that do not follow this convention are
 *    rejected with an error.
 *
 * A model directory looks like:
 *
 * ```
 * english-mfa-v3.1.0/
 *   meta.json       # feature chain + silence/OOV parameters
 *   final.alimdl    # alignment model (kaldi TransitionModel + AmDiagGmm)
 *   tree            # context dependency tree
 *   lda.mat         # LDA transform of the feature chain
 *   phones.txt      # phone table (optional)
 *   lexicon.txt     # pronunciation dictionary (see the lexicon field)
 *   en.wav          # test wav (added by the model package)
 *   README.md       # where the model comes from
 * ```
 *
 * Model packages ready to use are at
 * https://github.com/k2-fsa/sherpa-onnx/releases/tag/fa-models
 *
 * @property lexicon pronunciation dictionary, one pronunciation per line:
 *
 * ```
 * key [probability columns ...] phone1 phone2 ...
 * ```
 *
 * Probability columns (floating point numbers between the key and the
 * phones, e.g. MFA `.dict` files) are ignored. Several lines under one key
 * = several pronunciations of the same word/character, and ALL are kept:
 * each pronunciation becomes a parallel path in the lexicon FST and the
 * acoustics pick the one used by the alignment.
 *
 * The key shape depends on the language: English = words, Chinese = hanzi
 * characters/phrases (hanzi-keyed), other languages = their writing units.
 * Multi-word phrases keep their combined readings: a multi-token key
 * writes one `" || "`-separated phone group per word/character
 * (`银行 || i n || h a ŋ`); single-token keys need no `" || "`
 * (`行 x a ŋ`). Words without a pronunciation are reported as
 * out-of-vocabulary and aligned as the model's OOV phone.
 *
 * A lexicon looks like:
 *
 * ```
 * ask æ s k              # one word, one pronunciation
 * can k ə n              # a second word
 * can kʰ æ n             # a second pronunciation of the same word
 * 行 x a ŋ               # Chinese: one line per reading (polyphones)
 * 行 h a ŋ
 * 研究 || i n ɛ n || tɕ i o u  # per-character phone groups of a phrase
 * ```
 *
 * @property options default alignment options
 * @property debug true to print debug information: model info while loading
 *   the model and per-utterance alignment info in [ForcedAligner.compute]
 */
data class ForcedAlignerConfig(
    var modelDir: String = "",
    var lexicon: String = "",
    var options: ForcedAlignmentOptions = ForcedAlignmentOptions(),
    var debug: Boolean = false,
)
