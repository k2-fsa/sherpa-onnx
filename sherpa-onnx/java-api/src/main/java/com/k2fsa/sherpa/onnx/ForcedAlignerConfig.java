// Copyright (c)  2026  Xiaomi Corporation

package com.k2fsa.sherpa.onnx;

/**
 * Configuration for {@link ForcedAligner}: a forced-alignment acoustic
 * model directory, a pronunciation dictionary and the default alignment
 * options.
 */
public class ForcedAlignerConfig {
    /**
     * Directory of one forced-alignment acoustic model. Required files:
     *
     * <ul>
     *   <li>{@code meta.json}: feature chain + silence/OOV parameters
     *       (parsed once at construction, field by field)
     *   <li>{@code final.alimdl}: alignment model, kaldi format
     *       (TransitionModel + AmDiagGmm). Note: NOT {@code final.mdl}
     *   <li>{@code tree}: context dependency tree (ContextDependency)
     *   <li>{@code lda.mat}: LDA transform of the feature chain; required
     *       iff the feature chain ends in splice+LDA
     * </ul>
     *
     * <p>Optional file:
     *
     * <ul>
     *   <li>{@code phones.txt}: phone table (symbol to id). When missing,
     *       the phone ids are taken from the {@code phones} list of
     *       {@code meta.json} following the kaldi convention of these
     *       models ({@code <eps>=0}, {@code sil=1}, {@code spn=2}, then the
     *       lexical phones from 3). Models that do not follow this
     *       convention are rejected with an error.
     * </ul>
     *
     * <p>A model directory looks like:
     *
     * <pre>
     * english-mfa-v3.1.0/
     *   meta.json       # feature chain + silence/OOV parameters
     *   final.alimdl    # alignment model (kaldi TransitionModel + AmDiagGmm)
     *   tree            # context dependency tree
     *   lda.mat         # LDA transform of the feature chain
     *   phones.txt      # phone table (optional)
     *   lexicon.txt     # pronunciation dictionary (see the lexicon field)
     *   en.wav          # test wav (added by the model package)
     *   README.md       # where the model comes from
     * </pre>
     *
     * <p>Model packages ready to use are at
     * https://github.com/k2-fsa/sherpa-onnx/releases/tag/fa-models
     */
    public String modelDir;

    /**
     * Pronunciation dictionary, one pronunciation per line:
     *
     * <pre>
     * key [probability columns ...] phone1 phone2 ...
     * </pre>
     *
     * <p>Probability columns (floating point numbers between the key and
     * the phones, e.g. MFA {@code .dict} files) are ignored. Several lines
     * under one key = several pronunciations of the same word/character,
     * and ALL are kept: each pronunciation becomes a parallel path in the
     * lexicon FST and the acoustics pick the one used by the alignment.
     *
     * <p>The key shape depends on the language: English = words, Chinese =
     * hanzi characters/phrases (hanzi-keyed), other languages = their
     * writing units. Multi-word phrases keep their combined readings: a
     * multi-token key writes one {@code " || "}-separated phone group per
     * word/character ({@code 银行 || i n || h a ŋ}); single-token keys need
     * no {@code " || "} ({@code 行 x a ŋ}). Words without a pronunciation
     * are reported as out-of-vocabulary and aligned as the model's OOV
     * phone.
     *
     * <p>A lexicon looks like:
     *
     * <pre>
     * ask æ s k              # one word, one pronunciation
     * can k ə n              # a second word
     * can kʰ æ n             # a second pronunciation of the same word
     * 行 x a ŋ               # Chinese: one line per reading (polyphones)
     * 行 h a ŋ
     * 研究 || i n ɛ n || tɕ i o u  # per-character phone groups of a phrase
     * </pre>
     */
    public String lexicon;

    /** Default alignment options. */
    public ForcedAlignmentOptions options = new ForcedAlignmentOptions();

    /**
     * True to print debug information: model info while loading the model
     * and per-utterance alignment info in {@link ForcedAligner#compute}.
     */
    public boolean debug;
}
