//! Offline text-to-speech.
//!
//! Supported model families include VITS, Matcha, Kokoro, Kitten, ZipVoice,
//! Pocket TTS, and Supertonic. See the repository examples:
//!
//! - `rust-api-examples/examples/pocket_tts.rs`
//! - `rust-api-examples/examples/kokoro_tts_en.rs`
//! - `rust-api-examples/examples/kokoro_tts_zh_en.rs`
//! - `rust-api-examples/examples/matcha_tts_en.rs`
//! - `rust-api-examples/examples/matcha_tts_zh.rs`
//! - `rust-api-examples/examples/zipvoice_tts.rs`
//! - `rust-api-examples/examples/supertonic_tts.rs`
//!
//! # Example
//!
//! ```no_run
//! use sherpa_onnx::{
//!     GenerationConfig, OfflineTts, OfflineTtsConfig, OfflineTtsModelConfig,
//!     OfflineTtsPocketModelConfig, Wave,
//! };
//!
//! let config = OfflineTtsConfig {
//!     model: OfflineTtsModelConfig {
//!         pocket: OfflineTtsPocketModelConfig {
//!             lm_flow: Some("./sherpa-onnx-pocket-tts-int8-2026-01-26/lm_flow.int8.onnx".into()),
//!             lm_main: Some("./sherpa-onnx-pocket-tts-int8-2026-01-26/lm_main.int8.onnx".into()),
//!             encoder: Some("./sherpa-onnx-pocket-tts-int8-2026-01-26/encoder.onnx".into()),
//!             decoder: Some("./sherpa-onnx-pocket-tts-int8-2026-01-26/decoder.int8.onnx".into()),
//!             text_conditioner: Some(
//!                 "./sherpa-onnx-pocket-tts-int8-2026-01-26/text_conditioner.onnx".into(),
//!             ),
//!             vocab_json: Some("./sherpa-onnx-pocket-tts-int8-2026-01-26/vocab.json".into()),
//!             token_scores_json: Some(
//!                 "./sherpa-onnx-pocket-tts-int8-2026-01-26/token_scores.json".into(),
//!             ),
//!             ..Default::default()
//!         },
//!         ..Default::default()
//!     },
//!     ..Default::default()
//! };
//!
//! let tts = OfflineTts::create(&config).expect("create tts");
//! let reference = Wave::read("./sherpa-onnx-pocket-tts-int8-2026-01-26/test_wavs/bria.wav")
//!     .expect("read reference");
//! let generation_config = GenerationConfig {
//!     reference_audio: Some(reference.samples().to_vec()),
//!     reference_sample_rate: reference.sample_rate(),
//!     ..Default::default()
//! };
//! let audio = tts
//!     .generate_with_config("Hello from sherpa-onnx", &generation_config, None)
//!     .expect("generate");
//! println!("{}", audio.sample_rate());
//! ```

use crate::utils::to_c_ptr;
use sherpa_onnx_sys as sys;
use std::collections::HashMap;
use std::ffi::CString;
use std::os::raw::c_void;
use std::ptr;
use std::slice;

type ProgressCallback = dyn FnMut(&[f32], f32) -> bool;
type BoxedProgressCallback = Box<ProgressCallback>;

// --- Model config structs ---

#[derive(Clone, Debug)]
/// VITS model configuration.
pub struct OfflineTtsVitsModelConfig {
    pub model: Option<String>,
    pub lexicon: Option<String>,
    pub tokens: Option<String>,
    pub data_dir: Option<String>,
    pub noise_scale: f32,
    pub noise_scale_w: f32,
    pub length_scale: f32,
    pub dict_dir: Option<String>,
}

impl Default for OfflineTtsVitsModelConfig {
    fn default() -> Self {
        Self {
            model: None,
            lexicon: None,
            tokens: None,
            data_dir: None,
            noise_scale: 0.667,
            noise_scale_w: 0.8,
            length_scale: 1.0,
            dict_dir: None,
        }
    }
}

impl OfflineTtsVitsModelConfig {
    fn to_sys(&self, cstrings: &mut Vec<CString>) -> sys::OfflineTtsVitsModelConfig {
        sys::OfflineTtsVitsModelConfig {
            model: to_c_ptr(&self.model, cstrings),
            lexicon: to_c_ptr(&self.lexicon, cstrings),
            tokens: to_c_ptr(&self.tokens, cstrings),
            data_dir: to_c_ptr(&self.data_dir, cstrings),
            noise_scale: self.noise_scale,
            noise_scale_w: self.noise_scale_w,
            length_scale: self.length_scale,
            dict_dir: to_c_ptr(&self.dict_dir, cstrings),
        }
    }
}

#[derive(Clone, Debug)]
/// Matcha model configuration.
pub struct OfflineTtsMatchaModelConfig {
    pub acoustic_model: Option<String>,
    pub vocoder: Option<String>,
    pub lexicon: Option<String>,
    pub tokens: Option<String>,
    pub data_dir: Option<String>,
    pub noise_scale: f32,
    pub length_scale: f32,
    pub dict_dir: Option<String>,
}

impl Default for OfflineTtsMatchaModelConfig {
    fn default() -> Self {
        Self {
            acoustic_model: None,
            vocoder: None,
            lexicon: None,
            tokens: None,
            data_dir: None,
            noise_scale: 0.667,
            length_scale: 1.0,
            dict_dir: None,
        }
    }
}

impl OfflineTtsMatchaModelConfig {
    fn to_sys(&self, cstrings: &mut Vec<CString>) -> sys::OfflineTtsMatchaModelConfig {
        sys::OfflineTtsMatchaModelConfig {
            acoustic_model: to_c_ptr(&self.acoustic_model, cstrings),
            vocoder: to_c_ptr(&self.vocoder, cstrings),
            lexicon: to_c_ptr(&self.lexicon, cstrings),
            tokens: to_c_ptr(&self.tokens, cstrings),
            data_dir: to_c_ptr(&self.data_dir, cstrings),
            noise_scale: self.noise_scale,
            length_scale: self.length_scale,
            dict_dir: to_c_ptr(&self.dict_dir, cstrings),
        }
    }
}

#[derive(Clone, Debug)]
/// Kokoro model configuration.
pub struct OfflineTtsKokoroModelConfig {
    pub model: Option<String>,
    pub voices: Option<String>,
    pub tokens: Option<String>,
    pub data_dir: Option<String>,
    pub length_scale: f32,
    pub dict_dir: Option<String>,
    pub lexicon: Option<String>,
    pub lang: Option<String>,
}

impl Default for OfflineTtsKokoroModelConfig {
    fn default() -> Self {
        Self {
            model: None,
            voices: None,
            tokens: None,
            data_dir: None,
            length_scale: 1.0,
            dict_dir: None,
            lexicon: None,
            lang: None,
        }
    }
}

impl OfflineTtsKokoroModelConfig {
    fn to_sys(&self, cstrings: &mut Vec<CString>) -> sys::OfflineTtsKokoroModelConfig {
        sys::OfflineTtsKokoroModelConfig {
            model: to_c_ptr(&self.model, cstrings),
            voices: to_c_ptr(&self.voices, cstrings),
            tokens: to_c_ptr(&self.tokens, cstrings),
            data_dir: to_c_ptr(&self.data_dir, cstrings),
            length_scale: self.length_scale,
            dict_dir: to_c_ptr(&self.dict_dir, cstrings),
            lexicon: to_c_ptr(&self.lexicon, cstrings),
            lang: to_c_ptr(&self.lang, cstrings),
        }
    }
}

#[derive(Clone, Debug)]
/// Kitten model configuration.
pub struct OfflineTtsKittenModelConfig {
    pub model: Option<String>,
    pub voices: Option<String>,
    pub tokens: Option<String>,
    pub data_dir: Option<String>,
    pub length_scale: f32,
}

impl Default for OfflineTtsKittenModelConfig {
    fn default() -> Self {
        Self {
            model: None,
            voices: None,
            tokens: None,
            data_dir: None,
            length_scale: 1.0,
        }
    }
}

impl OfflineTtsKittenModelConfig {
    fn to_sys(&self, cstrings: &mut Vec<CString>) -> sys::OfflineTtsKittenModelConfig {
        sys::OfflineTtsKittenModelConfig {
            model: to_c_ptr(&self.model, cstrings),
            voices: to_c_ptr(&self.voices, cstrings),
            tokens: to_c_ptr(&self.tokens, cstrings),
            data_dir: to_c_ptr(&self.data_dir, cstrings),
            length_scale: self.length_scale,
        }
    }
}

#[derive(Clone, Debug)]
/// ZipVoice model configuration.
pub struct OfflineTtsZipvoiceModelConfig {
    pub tokens: Option<String>,
    pub encoder: Option<String>,
    pub decoder: Option<String>,
    pub vocoder: Option<String>,
    pub data_dir: Option<String>,
    pub lexicon: Option<String>,
    pub feat_scale: f32,
    pub t_shift: f32,
    pub target_rms: f32,
    pub guidance_scale: f32,
}

impl Default for OfflineTtsZipvoiceModelConfig {
    fn default() -> Self {
        Self {
            tokens: None,
            encoder: None,
            decoder: None,
            vocoder: None,
            data_dir: None,
            lexicon: None,
            feat_scale: 0.0,
            t_shift: 0.0,
            target_rms: 0.0,
            guidance_scale: 0.0,
        }
    }
}

impl OfflineTtsZipvoiceModelConfig {
    fn to_sys(&self, cstrings: &mut Vec<CString>) -> sys::OfflineTtsZipvoiceModelConfig {
        sys::OfflineTtsZipvoiceModelConfig {
            tokens: to_c_ptr(&self.tokens, cstrings),
            encoder: to_c_ptr(&self.encoder, cstrings),
            decoder: to_c_ptr(&self.decoder, cstrings),
            vocoder: to_c_ptr(&self.vocoder, cstrings),
            data_dir: to_c_ptr(&self.data_dir, cstrings),
            lexicon: to_c_ptr(&self.lexicon, cstrings),
            feat_scale: self.feat_scale,
            t_shift: self.t_shift,
            target_rms: self.target_rms,
            guidance_scale: self.guidance_scale,
        }
    }
}

#[derive(Clone, Debug, Default)]
/// Pocket TTS model configuration.
pub struct OfflineTtsPocketModelConfig {
    pub lm_flow: Option<String>,
    pub lm_main: Option<String>,
    pub encoder: Option<String>,
    pub decoder: Option<String>,
    pub text_conditioner: Option<String>,
    pub vocab_json: Option<String>,
    pub token_scores_json: Option<String>,
    pub voice_embedding_cache_capacity: i32,
}

impl OfflineTtsPocketModelConfig {
    fn to_sys(&self, cstrings: &mut Vec<CString>) -> sys::OfflineTtsPocketModelConfig {
        sys::OfflineTtsPocketModelConfig {
            lm_flow: to_c_ptr(&self.lm_flow, cstrings),
            lm_main: to_c_ptr(&self.lm_main, cstrings),
            encoder: to_c_ptr(&self.encoder, cstrings),
            decoder: to_c_ptr(&self.decoder, cstrings),
            text_conditioner: to_c_ptr(&self.text_conditioner, cstrings),
            vocab_json: to_c_ptr(&self.vocab_json, cstrings),
            token_scores_json: to_c_ptr(&self.token_scores_json, cstrings),
            voice_embedding_cache_capacity: self.voice_embedding_cache_capacity,
        }
    }
}

#[derive(Clone, Debug, Default)]
/// Supertonic model configuration.
pub struct OfflineTtsSupertonicModelConfig {
    pub duration_predictor: Option<String>,
    pub text_encoder: Option<String>,
    pub vector_estimator: Option<String>,
    pub vocoder: Option<String>,
    pub tts_json: Option<String>,
    pub unicode_indexer: Option<String>,
    pub voice_style: Option<String>,
}

impl OfflineTtsSupertonicModelConfig {
    fn to_sys(&self, cstrings: &mut Vec<CString>) -> sys::OfflineTtsSupertonicModelConfig {
        sys::OfflineTtsSupertonicModelConfig {
            duration_predictor: to_c_ptr(&self.duration_predictor, cstrings),
            text_encoder: to_c_ptr(&self.text_encoder, cstrings),
            vector_estimator: to_c_ptr(&self.vector_estimator, cstrings),
            vocoder: to_c_ptr(&self.vocoder, cstrings),
            tts_json: to_c_ptr(&self.tts_json, cstrings),
            unicode_indexer: to_c_ptr(&self.unicode_indexer, cstrings),
            voice_style: to_c_ptr(&self.voice_style, cstrings),
        }
    }
}

// --- Aggregate config structs ---

#[derive(Clone, Debug, Default)]
/// Aggregate model configuration for [`OfflineTts`].
///
/// Configure exactly one model family for typical use.
pub struct OfflineTtsModelConfig {
    pub vits: OfflineTtsVitsModelConfig,
    pub matcha: OfflineTtsMatchaModelConfig,
    pub kokoro: OfflineTtsKokoroModelConfig,
    pub kitten: OfflineTtsKittenModelConfig,
    pub zipvoice: OfflineTtsZipvoiceModelConfig,
    pub pocket: OfflineTtsPocketModelConfig,
    pub supertonic: OfflineTtsSupertonicModelConfig,
    pub num_threads: i32,
    pub debug: bool,
    pub provider: Option<String>,
}

impl OfflineTtsModelConfig {
    fn to_sys(&self, cstrings: &mut Vec<CString>) -> sys::OfflineTtsModelConfig {
        sys::OfflineTtsModelConfig {
            vits: self
                .vits
                .to_sys(cstrings),
            num_threads: self.num_threads,
            debug: self.debug as i32,
            provider: to_c_ptr(&self.provider, cstrings),
            matcha: self
                .matcha
                .to_sys(cstrings),
            kokoro: self
                .kokoro
                .to_sys(cstrings),
            kitten: self
                .kitten
                .to_sys(cstrings),
            zipvoice: self
                .zipvoice
                .to_sys(cstrings),
            pocket: self
                .pocket
                .to_sys(cstrings),
            supertonic: self
                .supertonic
                .to_sys(cstrings),
        }
    }
}

#[derive(Clone, Debug, Default)]
/// Top-level configuration for [`OfflineTts`].
pub struct OfflineTtsConfig {
    pub model: OfflineTtsModelConfig,
    pub rule_fsts: Option<String>,
    pub max_num_sentences: i32,
    pub rule_fars: Option<String>,
    pub silence_scale: f32,
}

impl OfflineTtsConfig {
    fn to_sys(&self, cstrings: &mut Vec<CString>) -> sys::OfflineTtsConfig {
        sys::OfflineTtsConfig {
            model: self
                .model
                .to_sys(cstrings),
            rule_fsts: to_c_ptr(&self.rule_fsts, cstrings),
            max_num_sentences: self.max_num_sentences,
            rule_fars: to_c_ptr(&self.rule_fars, cstrings),
            silence_scale: self.silence_scale,
        }
    }
}

// --- Generation config ---

#[derive(Clone, Debug)]
/// Per-request generation options for [`OfflineTts::generate_with_config`].
pub struct GenerationConfig {
    pub silence_scale: f32,
    pub speed: f32,
    pub sid: i32,
    pub reference_audio: Option<Vec<f32>>,
    pub reference_sample_rate: i32,
    pub reference_text: Option<String>,
    pub num_steps: i32,
    pub extra: Option<HashMap<String, serde_json::Value>>,
}

impl Default for GenerationConfig {
    fn default() -> Self {
        Self {
            silence_scale: 0.2,
            speed: 1.0,
            sid: 0,
            reference_audio: None,
            reference_sample_rate: 0,
            reference_text: None,
            num_steps: 5,
            extra: None,
        }
    }
}

// --- Generated audio ---

/// Generated audio returned by [`OfflineTts::generate_with_config`].
pub struct GeneratedAudio {
    ptr: *const sys::SherpaOnnxGeneratedAudio,
}

impl GeneratedAudio {
    /// Borrow generated samples.
    pub fn samples(&self) -> &[f32] {
        unsafe {
            let p = &*self.ptr;
            if p.samples
                .is_null()
                || p.n <= 0
            {
                &[]
            } else {
                slice::from_raw_parts(p.samples, p.n as usize)
            }
        }
    }

    /// Return the output sample rate in Hz.
    pub fn sample_rate(&self) -> i32 {
        unsafe { (*self.ptr).sample_rate }
    }

    /// Save generated audio to a WAV file.
    pub fn save(&self, filename: &str) -> bool {
        crate::wave::write(filename, self.samples(), self.sample_rate())
    }
}

impl Drop for GeneratedAudio {
    fn drop(&mut self) {
        unsafe {
            if !self
                .ptr
                .is_null()
            {
                sys::SherpaOnnxDestroyOfflineTtsGeneratedAudio(self.ptr);
            }
        }
    }
}

// --- Offline TTS ---

/// Offline TTS engine.
///
/// ```no_run
/// use sherpa_onnx::{
///     OfflineTts, OfflineTtsConfig, OfflineTtsModelConfig, OfflineTtsPocketModelConfig,
/// };
///
/// let config = OfflineTtsConfig {
///     model: OfflineTtsModelConfig {
///         pocket: OfflineTtsPocketModelConfig {
///             lm_flow: Some("./sherpa-onnx-pocket-tts-int8-2026-01-26/lm_flow.int8.onnx".into()),
///             lm_main: Some("./sherpa-onnx-pocket-tts-int8-2026-01-26/lm_main.int8.onnx".into()),
///             encoder: Some("./sherpa-onnx-pocket-tts-int8-2026-01-26/encoder.onnx".into()),
///             decoder: Some("./sherpa-onnx-pocket-tts-int8-2026-01-26/decoder.int8.onnx".into()),
///             text_conditioner: Some(
///                 "./sherpa-onnx-pocket-tts-int8-2026-01-26/text_conditioner.onnx".into(),
///             ),
///             vocab_json: Some("./sherpa-onnx-pocket-tts-int8-2026-01-26/vocab.json".into()),
///             token_scores_json: Some(
///                 "./sherpa-onnx-pocket-tts-int8-2026-01-26/token_scores.json".into(),
///             ),
///             ..Default::default()
///         },
///         ..Default::default()
///     },
///     ..Default::default()
/// };
///
/// let tts = OfflineTts::create(&config).expect("create tts");
/// println!("{}", tts.sample_rate());
/// ```
pub struct OfflineTts {
    ptr: *const sys::SherpaOnnxOfflineTts,
}

unsafe impl Send for OfflineTts {}

impl OfflineTts {
    /// Create a TTS engine from `config`.
    pub fn create(config: &OfflineTtsConfig) -> Option<Self> {
        let mut cstrings = Vec::new();
        let sys_config = config.to_sys(&mut cstrings);
        let ptr = unsafe { sys::SherpaOnnxCreateOfflineTts(&sys_config) };
        if ptr.is_null() {
            None
        } else {
            Some(Self { ptr })
        }
    }

    /// Return the output sample rate in Hz.
    pub fn sample_rate(&self) -> i32 {
        unsafe { sys::SherpaOnnxOfflineTtsSampleRate(self.ptr) }
    }

    /// Return the number of built-in speakers reported by the model.
    pub fn num_speakers(&self) -> i32 {
        unsafe { sys::SherpaOnnxOfflineTtsNumSpeakers(self.ptr) }
    }

    /// Generate audio for `text`.
    ///
    /// The optional callback receives the samples generated so far together
    /// with a progress value in `[0, 1]`. Return `true` to continue and
    /// `false` to stop early.
    pub fn generate_with_config<F>(
        &self,
        text: &str,
        config: &GenerationConfig,
        callback: Option<F>,
    ) -> Option<GeneratedAudio>
    where
        F: FnMut(&[f32], f32) -> bool + 'static,
    {
        let mut cstrings = Vec::new();

        let c_text = CString::new(text).unwrap();

        // Build extra JSON string
        let extra_json = match &config.extra {
            Some(map) => serde_json::to_string(map).unwrap_or_else(|_| "{}".to_string()),
            None => "{}".to_string(),
        };
        let c_extra = CString::new(extra_json).unwrap();
        let c_ref_text = to_c_ptr(&config.reference_text, &mut cstrings);

        let (ref_ptr, ref_len) = match &config.reference_audio {
            Some(samples) => (samples.as_ptr(), samples.len() as i32),
            None => (ptr::null(), 0),
        };

        let sys_gen_config = sys::SherpaOnnxGenerationConfig {
            silence_scale: config.silence_scale,
            speed: config.speed,
            sid: config.sid,
            reference_audio: ref_ptr,
            reference_audio_len: ref_len,
            reference_sample_rate: config.reference_sample_rate,
            reference_text: c_ref_text,
            num_steps: config.num_steps,
            extra: c_extra.as_ptr(),
        };

        let (c_callback, c_arg): (
            sys::SherpaOnnxGeneratedAudioProgressCallbackWithArg,
            *mut c_void,
        ) = if let Some(cb) = callback {
            let boxed: Box<BoxedProgressCallback> = Box::new(Box::new(cb));
            let raw = Box::into_raw(boxed);
            (Some(progress_callback_trampoline), raw as *mut c_void)
        } else {
            (None, ptr::null_mut())
        };

        let audio_ptr = unsafe {
            sys::SherpaOnnxOfflineTtsGenerateWithConfig(
                self.ptr,
                c_text.as_ptr(),
                &sys_gen_config,
                c_callback,
                c_arg,
            )
        };

        // Clean up the boxed callback if we allocated one
        if !c_arg.is_null() {
            unsafe {
                let _ = Box::from_raw(c_arg as *mut BoxedProgressCallback);
            }
        }

        if audio_ptr.is_null() {
            None
        } else {
            Some(GeneratedAudio { ptr: audio_ptr })
        }
    }
}

impl Drop for OfflineTts {
    fn drop(&mut self) {
        unsafe {
            sys::SherpaOnnxDestroyOfflineTts(self.ptr);
        }
    }
}

unsafe extern "C" fn progress_callback_trampoline(
    samples: *const f32,
    n: i32,
    progress: f32,
    arg: *mut c_void,
) -> i32 {
    let cb = &mut *(arg as *mut BoxedProgressCallback);
    let data = if samples.is_null() || n <= 0 {
        &[]
    } else {
        slice::from_raw_parts(samples, n as usize)
    };
    if cb(data, progress) {
        1
    } else {
        0
    }
}
