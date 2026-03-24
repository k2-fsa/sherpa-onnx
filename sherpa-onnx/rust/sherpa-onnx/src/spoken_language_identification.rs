//! Spoken language identification.
//!
//! This module identifies the language spoken in an audio clip using the
//! Whisper-based language ID API. See
//! [`rust-api-examples/examples/spoken_language_identification.rs`](https://github.com/k2-fsa/sherpa-onnx/blob/master/rust-api-examples/examples/spoken_language_identification.rs)
//! for a complete example.
//!
//! # Example
//!
//! ```no_run
//! use sherpa_onnx::{
//!     SpokenLanguageIdentification, SpokenLanguageIdentificationConfig,
//!     SpokenLanguageIdentificationWhisperConfig, Wave,
//! };
//!
//! let wave = Wave::read("./test.wav").expect("read wave");
//! let config = SpokenLanguageIdentificationConfig {
//!     whisper: SpokenLanguageIdentificationWhisperConfig {
//!         encoder: Some("./sherpa-onnx-whisper-tiny/encoder.int8.onnx".into()),
//!         decoder: Some("./sherpa-onnx-whisper-tiny/decoder.int8.onnx".into()),
//!         tail_paddings: 0,
//!     },
//!     ..Default::default()
//! };
//!
//! let slid = SpokenLanguageIdentification::create(&config).expect("create");
//! let stream = slid.create_stream();
//! stream.accept_waveform(wave.sample_rate(), wave.samples());
//! let result = slid.compute(&stream).expect("compute");
//! println!("{}", result.lang);
//! ```

use crate::offline_asr::OfflineStream;
use crate::utils::to_c_ptr;
use sherpa_onnx_sys as sys;
use std::ffi::{CStr, CString};

#[derive(Clone, Debug, Default)]
/// Whisper model configuration for spoken language identification.
pub struct SpokenLanguageIdentificationWhisperConfig {
    pub encoder: Option<String>,
    pub decoder: Option<String>,
    pub tail_paddings: i32,
}

impl SpokenLanguageIdentificationWhisperConfig {
    fn to_sys(
        &self,
        cstrings: &mut Vec<CString>,
    ) -> sys::SpokenLanguageIdentificationWhisperConfig {
        sys::SpokenLanguageIdentificationWhisperConfig {
            encoder: to_c_ptr(&self.encoder, cstrings),
            decoder: to_c_ptr(&self.decoder, cstrings),
            tail_paddings: self.tail_paddings,
        }
    }
}

#[derive(Clone, Debug)]
/// Top-level configuration for [`SpokenLanguageIdentification`].
pub struct SpokenLanguageIdentificationConfig {
    pub whisper: SpokenLanguageIdentificationWhisperConfig,
    pub num_threads: i32,
    pub debug: bool,
    pub provider: Option<String>,
}

impl Default for SpokenLanguageIdentificationConfig {
    fn default() -> Self {
        Self {
            whisper: Default::default(),
            num_threads: 1,
            debug: false,
            provider: Some("cpu".to_string()),
        }
    }
}

impl SpokenLanguageIdentificationConfig {
    fn to_sys(&self, cstrings: &mut Vec<CString>) -> sys::SpokenLanguageIdentificationConfig {
        sys::SpokenLanguageIdentificationConfig {
            whisper: self
                .whisper
                .to_sys(cstrings),
            num_threads: self.num_threads,
            debug: self.debug as i32,
            provider: to_c_ptr(&self.provider, cstrings),
        }
    }
}

#[derive(Clone, Debug)]
/// Result returned by [`SpokenLanguageIdentification::compute`].
pub struct SpokenLanguageIdentificationResult {
    pub lang: String,
}

/// Spoken language identifier.
pub struct SpokenLanguageIdentification {
    ptr: *const sys::SpokenLanguageIdentification,
}

unsafe impl Send for SpokenLanguageIdentification {}

impl SpokenLanguageIdentification {
    /// Create a language identifier from [`SpokenLanguageIdentificationConfig`].
    pub fn create(config: &SpokenLanguageIdentificationConfig) -> Option<Self> {
        let mut cstrings = Vec::new();
        let sys_config = config.to_sys(&mut cstrings);
        let ptr = unsafe { sys::SherpaOnnxCreateSpokenLanguageIdentification(&sys_config) };
        if ptr.is_null() {
            None
        } else {
            Some(Self { ptr })
        }
    }

    /// Create an offline stream for one audio clip.
    pub fn create_stream(&self) -> OfflineStream {
        let ptr =
            unsafe { sys::SherpaOnnxSpokenLanguageIdentificationCreateOfflineStream(self.ptr) };
        OfflineStream { ptr }
    }

    /// Compute the spoken language for `stream`.
    pub fn compute(&self, stream: &OfflineStream) -> Option<SpokenLanguageIdentificationResult> {
        unsafe {
            let p = sys::SherpaOnnxSpokenLanguageIdentificationCompute(self.ptr, stream.ptr);
            if p.is_null() {
                return None;
            }

            let ans = SpokenLanguageIdentificationResult {
                lang: if (*p)
                    .lang
                    .is_null()
                {
                    String::new()
                } else {
                    CStr::from_ptr((*p).lang)
                        .to_string_lossy()
                        .into_owned()
                },
            };

            sys::SherpaOnnxDestroySpokenLanguageIdentificationResult(p);
            Some(ans)
        }
    }
}

impl Drop for SpokenLanguageIdentification {
    fn drop(&mut self) {
        unsafe {
            if !self
                .ptr
                .is_null()
            {
                sys::SherpaOnnxDestroySpokenLanguageIdentification(self.ptr);
            }
        }
    }
}
