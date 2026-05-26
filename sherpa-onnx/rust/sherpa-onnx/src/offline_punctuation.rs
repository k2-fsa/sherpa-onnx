//! Offline punctuation restoration.
//!
//! Use this module when you already have a complete text string and want a
//! one-shot punctuation pass. See
//! [`rust-api-examples/examples/offline_punctuation.rs`](https://github.com/k2-fsa/sherpa-onnx/blob/master/rust-api-examples/examples/offline_punctuation.rs)
//! for a complete example.
//!
//! # Example
//!
//! ```no_run
//! use sherpa_onnx::{OfflinePunctuation, OfflinePunctuationConfig};
//!
//! let mut config = OfflinePunctuationConfig::default();
//! config.model.ct_transformer = Some("./sherpa-onnx-offline-punctuation/model.onnx".into());
//!
//! let punct = OfflinePunctuation::create(&config).expect("create punctuator");
//! let text = punct
//!     .add_punctuation("today is a good day how are you")
//!     .expect("punctuate");
//! println!("{text}");
//! ```

use crate::utils::to_c_ptr;
use sherpa_onnx_sys as sys;
use std::ffi::{CStr, CString};

#[derive(Clone, Debug)]
/// Model configuration for offline punctuation restoration.
pub struct OfflinePunctuationModelConfig {
    pub ct_transformer: Option<String>,
    pub num_threads: i32,
    pub debug: bool,
    pub provider: Option<String>,
}

impl Default for OfflinePunctuationModelConfig {
    fn default() -> Self {
        Self {
            ct_transformer: None,
            num_threads: 1,
            debug: false,
            provider: Some("cpu".to_string()),
        }
    }
}

impl OfflinePunctuationModelConfig {
    fn to_sys(&self, cstrings: &mut Vec<CString>) -> sys::OfflinePunctuationModelConfig {
        sys::OfflinePunctuationModelConfig {
            ct_transformer: to_c_ptr(&self.ct_transformer, cstrings),
            num_threads: self.num_threads,
            debug: self.debug as i32,
            provider: to_c_ptr(&self.provider, cstrings),
        }
    }
}

#[derive(Clone, Debug, Default)]
/// Top-level configuration for [`OfflinePunctuation`].
pub struct OfflinePunctuationConfig {
    pub model: OfflinePunctuationModelConfig,
}

impl OfflinePunctuationConfig {
    fn to_sys(&self, cstrings: &mut Vec<CString>) -> sys::OfflinePunctuationConfig {
        sys::OfflinePunctuationConfig {
            model: self
                .model
                .to_sys(cstrings),
        }
    }
}

/// Offline punctuation restorer.
pub struct OfflinePunctuation {
    ptr: *const sys::OfflinePunctuation,
}

// SAFETY: The sherpa-onnx C library is thread-safe for single-object usage.
unsafe impl Send for OfflinePunctuation {}
unsafe impl Sync for OfflinePunctuation {}

impl OfflinePunctuation {
    /// Create an offline punctuator from [`OfflinePunctuationConfig`].
    pub fn create(config: &OfflinePunctuationConfig) -> Option<Self> {
        let mut cstrings = Vec::new();
        let sys_config = config.to_sys(&mut cstrings);
        let ptr = unsafe { sys::SherpaOnnxCreateOfflinePunctuation(&sys_config) };
        if ptr.is_null() {
            None
        } else {
            Some(Self { ptr })
        }
    }

    /// Add punctuation to `text`.
    pub fn add_punctuation(&self, text: &str) -> Option<String> {
        let text = CString::new(text).ok()?;

        unsafe {
            let p = sys::SherpaOfflinePunctuationAddPunct(self.ptr, text.as_ptr());
            if p.is_null() {
                return None;
            }

            let ans = CStr::from_ptr(p)
                .to_string_lossy()
                .into_owned();
            sys::SherpaOfflinePunctuationFreeText(p);
            Some(ans)
        }
    }
}

impl Drop for OfflinePunctuation {
    fn drop(&mut self) {
        unsafe {
            if !self
                .ptr
                .is_null()
            {
                sys::SherpaOnnxDestroyOfflinePunctuation(self.ptr);
            }
        }
    }
}
