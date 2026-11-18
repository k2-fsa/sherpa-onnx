#![allow(non_camel_case_types)]
#![allow(non_snake_case)]
#![allow(non_upper_case_globals)]

use std::os::raw::c_char;

use crate::offline_asr::OfflineStream;

/// Alignment options of a forced aligner; a value of 0 lets the C API use
/// its default for that option.
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct ForcedAlignmentOptions {
    /// Decoding beam used by the alignment search (0 means the default 10).
    pub beam: f32,

    /// Decoding beam used to retry failed alignments (0 means the default
    /// 40).
    pub retry_beam: f32,

    /// Transition-probability scale relative to the acoustics (0 means the
    /// default 1).
    pub transition_scale: f32,

    /// Scale of self-loop versus non-self-loop log probs relative to the
    /// acoustics (0 means the default 0.1).
    pub self_loop_scale: f32,

    /// Scaling factor for acoustic likelihoods (0 means the default 0.1).
    pub acoustic_scale: f32,
}

/// Configuration of a forced aligner.
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct ForcedAlignerConfig {
    /// Path to the model directory. It must contain meta.json,
    /// final.alimdl and tree; lda.mat and phones.txt are optional.
    pub model_dir: *const c_char,

    /// Path to the pronunciation dictionary used to build the lexicon FST.
    pub lexicon: *const c_char,

    /// Default alignment options.
    pub options: ForcedAlignmentOptions,

    /// Non-zero to print debug information.
    pub debug: i32,
}

/// One aligned interval: a phone in the "phones" tier or a word in the
/// "words" tier.
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct ForcedAlignmentInterval {
    /// Start time in seconds.
    pub start: f32,

    /// End time in seconds.
    pub end: f32,

    /// The label of this interval; empty for a silence gap in the "words"
    /// tier.
    pub text: *const c_char,
}

/// One tier of the alignment (phones or words).
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct ForcedAlignmentTier {
    /// Tier name: "phones" or "words".
    pub name: *const c_char,

    /// Number of entries in [`Self::intervals`].
    pub num_intervals: i32,

    /// The aligned intervals of this tier, in time order.
    pub intervals: *const ForcedAlignmentInterval,
}

/// Result of [`SherpaOnnxForcedAlignerCompute`].
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct ForcedAlignmentResult {
    /// Duration of the aligned audio in seconds.
    pub duration: f32,

    /// The whole alignment as a Praat TextGrid string; empty for an empty
    /// result.
    pub textgrid: *const c_char,

    /// Number of entries in [`Self::tiers`]; 0 for an empty result.
    pub num_tiers: i32,

    /// The alignment tiers: tiers[0] is "phones", tiers[1] is "words".
    pub tiers: *const ForcedAlignmentTier,
}

/// Opaque forced aligner object.
#[repr(C)]
pub struct ForcedAligner {
    _private: [u8; 0],
}

extern "C" {
    /// Create a forced aligner from a model directory and a pronunciation
    /// dictionary. Returns NULL on failure; free the result with
    /// [`SherpaOnnxDestroyForcedAligner`].
    pub fn SherpaOnnxCreateForcedAligner(
        config: *const ForcedAlignerConfig,
    ) -> *const ForcedAligner;

    /// Free a forced aligner created by
    /// [`SherpaOnnxCreateForcedAligner`].
    pub fn SherpaOnnxDestroyForcedAligner(aligner: *const ForcedAligner);

    /// Create an utterance stream for a forced aligner. Fill it with
    /// `SherpaOnnxAcceptWaveformOffline` and free it with
    /// `SherpaOnnxDestroyOfflineStream`.
    pub fn SherpaOnnxForcedAlignerCreateStream(
        aligner: *const ForcedAligner,
    ) -> *const OfflineStream;

    /// Compute the alignment of one utterance. Returns NULL if the
    /// alignment itself fails, and an empty result (no tiers, empty
    /// textgrid) for the recoverable cases; free the result with
    /// [`SherpaOnnxDestroyForcedAlignmentResult`].
    pub fn SherpaOnnxForcedAlignerCompute(
        aligner: *const ForcedAligner,
        stream: *const OfflineStream,
        text: *const c_char,
    ) -> *const ForcedAlignmentResult;

    /// Free the result of [`SherpaOnnxForcedAlignerCompute`].
    pub fn SherpaOnnxDestroyForcedAlignmentResult(
        result: *const ForcedAlignmentResult,
    );
}
