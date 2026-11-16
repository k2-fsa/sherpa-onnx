package com.k2fsa.sherpa.onnx

/**
 * Per-utterance alignment options, following kalpy's GmmAligner defaults.
 *
 * An instance of this class is the `options` field of [ForcedAlignerConfig].
 *
 * @property beam decoding beam of the first alignment attempt. Default: 10
 * @property retryBeam decoding beam of the second attempt when the first one
 *   fails. Default: 40
 * @property transitionScale scale applied to the transition probabilities of
 *   the HMM topology. Default: 1
 * @property selfLoopScale scale applied to the self-loop transition
 *   probabilities. Default: 0.1
 * @property acousticScale scale applied to the acoustic (GMM) log-likelihoods.
 *   Default: 0.1
 */
data class ForcedAlignmentOptions(
    var beam: Float = 10f,
    var retryBeam: Float = 40f,
    var transitionScale: Float = 1f,
    var selfLoopScale: Float = 0.1f,
    var acousticScale: Float = 0.1f,
)
