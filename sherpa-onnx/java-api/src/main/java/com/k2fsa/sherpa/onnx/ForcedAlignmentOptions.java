// Copyright (c)  2026  Xiaomi Corporation

package com.k2fsa.sherpa.onnx;

/**
 * Per-utterance alignment options, following kalpy's GmmAligner defaults.
 *
 * <p>An instance of this class is the {@code options} field of
 * {@link ForcedAlignerConfig}. The fields are public; set the ones you want
 * to override.
 */
public class ForcedAlignmentOptions {
    /** Decoding beam of the first alignment attempt. Default: 10. */
    public float beam = 10;

    /** Decoding beam of the second attempt when the first one fails.
     * Default: 40. */
    public float retryBeam = 40;

    /** Scale applied to the transition probabilities of the HMM topology.
     * Default: 1. */
    public float transitionScale = 1;

    /** Scale applied to the self-loop transition probabilities. Default:
     * 0.1. */
    public float selfLoopScale = 0.1f;

    /** Scale applied to the acoustic (GMM) log-likelihoods. Default: 0.1. */
    public float acousticScale = 0.1f;
}
