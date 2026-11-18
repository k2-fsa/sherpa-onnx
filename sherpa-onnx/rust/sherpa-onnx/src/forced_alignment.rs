//! Forced alignment.
//!
//! Given a wave and its transcript, a [`ForcedAligner`] computes the
//! start/end time of every phone and word, and the same content as a Praat
//! TextGrid. See
//! [`rust-api-examples/examples/forced_alignment_english.rs`](https://github.com/k2-fsa/sherpa-onnx/blob/master/rust-api-examples/examples/forced_alignment_english.rs)
//! for a complete example.
//!
//! # Example
//!
//! ```no_run
//! use sherpa_onnx::{ForcedAligner, ForcedAlignerConfig, Wave};
//!
//! let wave = Wave::read("./english-mfa-v3.1.0/en.wav").unwrap();
//!
//! let mut config = ForcedAlignerConfig::default();
//! config.model_dir = Some("./english-mfa-v3.1.0".into());
//! config.lexicon = Some("./english-mfa-v3.1.0/lexicon.txt".into());
//!
//! let aligner = ForcedAligner::create(&config).expect("create aligner");
//! let stream = aligner.create_stream();
//! stream.accept_waveform(wave.sample_rate(), wave.samples());
//!
//! let text = "ask not what your country can do for you, \
//!             ask what you can do for your country.";
//! let result = aligner.compute(&stream, text).expect("alignment");
//!
//! std::fs::write("en.TextGrid", &result.text_grid).unwrap();
//! for tier in &result.tiers {
//!     for interval in &tier.intervals {
//!         println!("{} - {}  {}", interval.start, interval.end, interval.text);
//!     }
//! }
//! ```

use crate::offline_asr::OfflineStream;
use crate::utils::to_c_ptr;
use sherpa_onnx_sys as sys;
use std::ffi::{CStr, CString};
use std::os::raw::c_char;

#[derive(Clone, Debug)]
/// Alignment options of a forced aligner. A stream can override any of
/// them before the alignment is computed.
pub struct ForcedAlignmentOptions {
    /// Decoding beam used by the alignment search. Larger is more accurate
    /// and slower. If set to 0, the C API uses the default value 10.
    pub beam: f32,

    /// Decoding beam used to retry failed alignments. If set to 0, the C
    /// API uses the default value 40.
    pub retry_beam: f32,

    /// Transition-probability scale relative to the acoustics. If set to
    /// 0, the C API uses the default value 1.
    pub transition_scale: f32,

    /// Scale of self-loop versus non-self-loop log probs relative to the
    /// acoustics. If set to 0, the C API uses the default value 0.1.
    pub self_loop_scale: f32,

    /// Scaling factor for acoustic likelihoods. If set to 0, the C API
    /// uses the default value 0.1.
    pub acoustic_scale: f32,
}

impl Default for ForcedAlignmentOptions {
    fn default() -> Self {
        Self {
            beam: 10.0,
            retry_beam: 40.0,
            transition_scale: 1.0,
            self_loop_scale: 0.1,
            acoustic_scale: 0.1,
        }
    }
}

#[derive(Clone, Debug, Default)]
/// Configuration of a [`ForcedAligner`].
pub struct ForcedAlignerConfig {
    /// Directory of one forced-alignment acoustic model. Required files:
    ///
    /// - `meta.json`: feature chain + silence/OOV parameters (parsed once
    ///   at construction, field by field)
    /// - `final.alimdl`: the alignment model (kaldi format:
    ///   TransitionModel + AmDiagGmm); note it is `final.alimdl`, NOT
    ///   `final.mdl`
    /// - `tree`: the context dependency tree (ContextDependency)
    /// - `lda.mat`: required iff the feature chain is splice+LDA
    /// - `phones.txt`: optional phone table (falls back to the phones list
    ///   in `meta.json` when missing)
    ///
    /// Other files of the upstream model bundles (`final.mdl`,
    /// `phone_lm.fst`, `graphemes.txt`, `rules.yaml`, ...) are unused and
    /// ignored; missing required files are reported by name.
    pub model_dir: Option<String>,

    /// Path to the pronunciation dictionary used to build the lexicon
    /// FST. One pronunciation per line:
    ///
    /// ```text
    /// key [probability columns ...] phone1 phone2 ...
    /// ```
    ///
    /// Probability columns (floating point numbers between the key and
    /// the phones, e.g. MFA `.dict` files) are ignored. Several lines
    /// under one key give several pronunciations of the same
    /// word/character and ALL are kept: each pronunciation becomes a
    /// parallel path in the lexicon FST and the acoustics pick the one
    /// used by the alignment. This is how Chinese polyphones avoid
    /// text-level guessing.
    ///
    /// The key shape depends on the language: English = words, Chinese =
    /// hanzi characters/phrases (hanzi-keyed), other languages = their
    /// writing units. Multi-word phrases keep their combined readings: a
    /// multi-token key writes one `" || "`-separated phone group per
    /// word/character (`银行 || i n || h a ŋ`); single-token keys need no
    /// `" || "` (`行 x a ŋ`). Words without a pronunciation are reported
    /// as out-of-vocabulary and aligned as the model's OOV phone.
    pub lexicon: Option<String>,

    /// Default alignment options; a stream can override any of them
    /// before the alignment is computed.
    pub options: ForcedAlignmentOptions,

    /// true to print debug information: model info while loading the
    /// model and per-utterance alignment info in [`ForcedAligner::compute`].
    pub debug: bool,
}

impl ForcedAlignerConfig {
    fn to_sys(&self, cstrings: &mut Vec<CString>) -> sys::ForcedAlignerConfig {
        sys::ForcedAlignerConfig {
            model_dir: to_c_ptr(&self.model_dir, cstrings),
            lexicon: to_c_ptr(&self.lexicon, cstrings),
            options: sys::ForcedAlignmentOptions {
                beam: self.options.beam,
                retry_beam: self.options.retry_beam,
                transition_scale: self.options.transition_scale,
                self_loop_scale: self.options.self_loop_scale,
                acoustic_scale: self.options.acoustic_scale,
            },
            debug: self.debug as i32,
        }
    }
}

#[derive(Clone, Debug)]
/// One aligned interval: a phone in the "phones" tier or a word in the
/// "words" tier.
pub struct AlignmentInterval {
    /// Start time in seconds.
    pub start: f32,

    /// End time in seconds.
    pub end: f32,

    /// The label of this interval: the word for a word interval, the
    /// phone symbol for a phone interval. It is empty for a silence gap in
    /// the "words" tier (Praat interval tiers must tile the full time
    /// range, so the gaps between words are reported as intervals with
    /// empty text).
    pub text: String,
}

#[derive(Clone, Debug)]
/// One tier of the alignment (phones or words).
pub struct AlignmentTier {
    /// Tier name: exactly "phones" or "words"; never empty.
    pub name: String,

    /// The aligned intervals of this tier, in time order.
    pub intervals: Vec<AlignmentInterval>,
}

#[derive(Clone, Debug)]
/// Result of [`ForcedAligner::compute`]: the phone and word boundaries of
/// one utterance, and the same content as a Praat TextGrid.
pub struct ForcedAlignmentResult {
    /// Duration of the aligned utterance in seconds.
    pub duration: f32,

    /// The alignment tiers: `tiers[0]` is "phones", `tiers[1]` is
    /// "words". Empty for an empty result.
    pub tiers: Vec<AlignmentTier>,

    /// The alignment as a Praat TextGrid string. Write it to a
    /// `.TextGrid` file and open it with Praat
    /// (https://www.fon.hum.uva.nl/praat/) to visualize the phone and
    /// word boundaries. Empty for an empty result.
    pub text_grid: String,
}

/// Forced aligner: given a wave and its transcript, it computes the
/// start/end time of every phone and word.
pub struct ForcedAligner {
    ptr: *const sys::ForcedAligner,
}

// SAFETY: The sherpa-onnx C library is thread-safe for single-object usage.
unsafe impl Send for ForcedAligner {}
unsafe impl Sync for ForcedAligner {}

impl ForcedAligner {
    /// Create a forced aligner from [`ForcedAlignerConfig`].
    pub fn create(config: &ForcedAlignerConfig) -> Option<Self> {
        let mut cstrings = Vec::new();
        let sys_config = config.to_sys(&mut cstrings);
        let ptr = unsafe { sys::SherpaOnnxCreateForcedAligner(&sys_config) };
        if ptr.is_null() {
            None
        } else {
            Some(Self { ptr })
        }
    }

    /// Create an utterance stream. Fill it with
    /// [`OfflineStream::accept_waveform`], then pass it to
    /// [`Self::compute`].
    pub fn create_stream(&self) -> OfflineStream {
        let ptr = unsafe { sys::SherpaOnnxForcedAlignerCreateStream(self.ptr) };
        OfflineStream { ptr }
    }

    /// Compute the alignment of one utterance.
    ///
    /// Returns `None` when the alignment itself fails. The documented
    /// empty result (empty text, no features, every word
    /// out-of-vocabulary, or word boundaries cannot be recovered) comes
    /// back as `Some(result)` with empty `tiers` and empty `text_grid`;
    /// the reason is printed to stderr in that case.
    pub fn compute(
        &self,
        stream: &OfflineStream,
        text: &str,
    ) -> Option<ForcedAlignmentResult> {
        let text = CString::new(text).ok()?;

        unsafe {
            let r =
                sys::SherpaOnnxForcedAlignerCompute(self.ptr, stream.ptr, text.as_ptr());
            if r.is_null() {
                return None;
            }

            let duration = (*r).duration;
            let text_grid = c_string_to_string((*r).textgrid);

            let tiers = if (*r).tiers.is_null() || (*r).num_tiers <= 0 {
                Vec::new()
            } else {
                let tiers =
                    std::slice::from_raw_parts((*r).tiers, (*r).num_tiers as usize);
                tiers
                    .iter()
                    .map(|t| AlignmentTier {
                        name: c_string_to_string(t.name),
                        intervals: if t.intervals.is_null() || t.num_intervals <= 0 {
                            Vec::new()
                        } else {
                            let intervals = std::slice::from_raw_parts(
                                t.intervals,
                                t.num_intervals as usize,
                            );
                            intervals
                                .iter()
                                .map(|iv| AlignmentInterval {
                                    start: iv.start,
                                    end: iv.end,
                                    text: c_string_to_string(iv.text),
                                })
                                .collect()
                        },
                    })
                    .collect()
            };

            sys::SherpaOnnxDestroyForcedAlignmentResult(r);

            Some(ForcedAlignmentResult {
                duration,
                tiers,
                text_grid,
            })
        }
    }
}

impl Drop for ForcedAligner {
    fn drop(&mut self) {
        unsafe {
            if !self
                .ptr
                .is_null()
            {
                sys::SherpaOnnxDestroyForcedAligner(self.ptr);
            }
        }
    }
}

fn c_string_to_string(ptr: *const c_char) -> String {
    if ptr.is_null() {
        return String::new();
    }

    unsafe {
        CStr::from_ptr(ptr)
            .to_string_lossy()
            .into_owned()
    }
}
