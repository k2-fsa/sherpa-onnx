//! Online punctuation restoration.
//!
//! This module wraps the punctuation model used in
//! `rust-api-examples/examples/online_punctuation.rs`.
//!
//! ```no_run
//! use sherpa_onnx::{OnlinePunctuation, OnlinePunctuationConfig, OnlinePunctuationModelConfig};
//!
//! let config = OnlinePunctuationConfig {
//!     model: OnlinePunctuationModelConfig {
//!         cnn_bilstm: Some("./sherpa-onnx-online-punct-en/cnn_bilstm.onnx".into()),
//!         bpe_vocab: Some("./sherpa-onnx-online-punct-en/bpe.vocab".into()),
//!         ..Default::default()
//!     },
//! };
//!
//! let punct = OnlinePunctuation::create(&config).expect("create punctuation");
//! let text = punct
//!     .add_punctuation("how are you i am fine thank you")
//!     .expect("punctuate");
//! println!("{text}");
//! ```

use crate::utils::to_c_ptr;
use sherpa_onnx_sys as sys;
use std::ffi::{CStr, CString};

#[derive(Clone, Debug)]
/// Model-level options for online punctuation restoration.
pub struct OnlinePunctuationModelConfig {
    pub cnn_bilstm: Option<String>,
    pub bpe_vocab: Option<String>,
    pub num_threads: i32,
    pub debug: bool,
    pub provider: Option<String>,
}

impl Default for OnlinePunctuationModelConfig {
    fn default() -> Self {
        Self {
            cnn_bilstm: None,
            bpe_vocab: None,
            num_threads: 1,
            debug: false,
            provider: Some("cpu".to_string()),
        }
    }
}

impl OnlinePunctuationModelConfig {
    fn to_sys(&self, cstrings: &mut Vec<CString>) -> sys::OnlinePunctuationModelConfig {
        sys::OnlinePunctuationModelConfig {
            cnn_bilstm: to_c_ptr(&self.cnn_bilstm, cstrings),
            bpe_vocab: to_c_ptr(&self.bpe_vocab, cstrings),
            num_threads: self.num_threads,
            debug: self.debug as i32,
            provider: to_c_ptr(&self.provider, cstrings),
        }
    }
}

#[derive(Clone, Debug, Default)]
/// Top-level configuration for [`OnlinePunctuation`].
pub struct OnlinePunctuationConfig {
    pub model: OnlinePunctuationModelConfig,
}

impl OnlinePunctuationConfig {
    fn to_sys(&self, cstrings: &mut Vec<CString>) -> sys::OnlinePunctuationConfig {
        sys::OnlinePunctuationConfig {
            model: self
                .model
                .to_sys(cstrings),
        }
    }
}

/// Online punctuation restorer.
///
/// Feed plain text fragments to [`OnlinePunctuation::add_punctuation`] and get
/// punctuated text back.
pub struct OnlinePunctuation {
    ptr: *const sys::OnlinePunctuation,
}

unsafe impl Send for OnlinePunctuation {}

impl OnlinePunctuation {
    pub fn create(config: &OnlinePunctuationConfig) -> Option<Self> {
        let mut cstrings = Vec::new();
        let sys_config = config.to_sys(&mut cstrings);

        let ptr = unsafe { sys::SherpaOnnxCreateOnlinePunctuation(&sys_config) };
        if ptr.is_null() {
            None
        } else {
            Some(Self { ptr })
        }
    }

    /// Add punctuation to a text fragment.
    ///
    /// Returns `None` if the input cannot be converted to a C string or the
    /// native punctuator fails.
    pub fn add_punctuation(&self, text: &str) -> Option<String> {
        let text = CString::new(text).ok()?;

        unsafe {
            let p = sys::SherpaOnnxOnlinePunctuationAddPunct(self.ptr, text.as_ptr());
            if p.is_null() {
                return None;
            }

            let ans = CStr::from_ptr(p)
                .to_string_lossy()
                .into_owned();
            sys::SherpaOnnxOnlinePunctuationFreeText(p);
            Some(ans)
        }
    }
}

impl Drop for OnlinePunctuation {
    fn drop(&mut self) {
        unsafe {
            if !self
                .ptr
                .is_null()
            {
                sys::SherpaOnnxDestroyOnlinePunctuation(self.ptr);
            }
        }
    }
}
