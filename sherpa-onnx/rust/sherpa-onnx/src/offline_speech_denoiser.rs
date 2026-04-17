//! Offline speech denoising.
//!
//! Supported model families mirror the native API and currently include GTCRN
//! and DPDFNet. See the repository examples:
//!
//! - `rust-api-examples/examples/offline_speech_enhancement_gtcrn.rs`
//! - `rust-api-examples/examples/offline_speech_enhancement_dpdfnet.rs`

use crate::utils::to_c_ptr;
use sherpa_onnx_sys as sys;
use std::ffi::CString;
use std::ptr;
use std::slice;

#[derive(Clone, Debug, Default)]
/// GTCRN model path for offline denoising.
pub struct OfflineSpeechDenoiserGtcrnModelConfig {
    pub model: Option<String>,
}

impl OfflineSpeechDenoiserGtcrnModelConfig {
    pub(crate) fn to_sys(
        &self,
        cstrings: &mut Vec<CString>,
    ) -> sys::OfflineSpeechDenoiserGtcrnModelConfig {
        sys::OfflineSpeechDenoiserGtcrnModelConfig {
            model: to_c_ptr(&self.model, cstrings),
        }
    }
}

#[derive(Clone, Debug, Default)]
/// DPDFNet model path for offline denoising.
pub struct OfflineSpeechDenoiserDpdfNetModelConfig {
    pub model: Option<String>,
}

impl OfflineSpeechDenoiserDpdfNetModelConfig {
    pub(crate) fn to_sys(
        &self,
        cstrings: &mut Vec<CString>,
    ) -> sys::OfflineSpeechDenoiserDpdfNetModelConfig {
        sys::OfflineSpeechDenoiserDpdfNetModelConfig {
            model: to_c_ptr(&self.model, cstrings),
        }
    }
}

#[derive(Clone, Debug)]
/// Aggregate model configuration for [`OfflineSpeechDenoiser`].
///
/// Configure exactly one model family in normal use.
pub struct OfflineSpeechDenoiserModelConfig {
    pub gtcrn: OfflineSpeechDenoiserGtcrnModelConfig,
    pub dpdfnet: OfflineSpeechDenoiserDpdfNetModelConfig,
    pub num_threads: i32,
    pub debug: bool,
    pub provider: Option<String>,
}

impl Default for OfflineSpeechDenoiserModelConfig {
    fn default() -> Self {
        Self {
            gtcrn: Default::default(),
            dpdfnet: Default::default(),
            num_threads: 1,
            debug: false,
            provider: Some("cpu".to_string()),
        }
    }
}

impl OfflineSpeechDenoiserModelConfig {
    pub(crate) fn to_sys(
        &self,
        cstrings: &mut Vec<CString>,
    ) -> sys::OfflineSpeechDenoiserModelConfig {
        sys::OfflineSpeechDenoiserModelConfig {
            gtcrn: self
                .gtcrn
                .to_sys(cstrings),
            num_threads: self.num_threads,
            debug: self.debug as i32,
            provider: to_c_ptr(&self.provider, cstrings),
            dpdfnet: self
                .dpdfnet
                .to_sys(cstrings),
        }
    }
}

#[derive(Clone, Debug, Default)]
/// Denoised samples returned from an offline or online denoiser.
pub struct DenoisedAudio {
    pub samples: Vec<f32>,
    pub sample_rate: i32,
}

impl DenoisedAudio {
    pub(crate) fn from_ptr(ptr: *const sys::DenoisedAudio) -> Self {
        if ptr.is_null() {
            return Self::default();
        }

        unsafe {
            let n = (*ptr)
                .n
                .max(0) as usize;
            let samples = if (*ptr)
                .samples
                .is_null()
                || n == 0
            {
                vec![]
            } else {
                slice::from_raw_parts((*ptr).samples, n).to_vec()
            };
            let sample_rate = (*ptr).sample_rate;
            sys::SherpaOnnxDestroyDenoisedAudio(ptr);
            Self {
                samples,
                sample_rate,
            }
        }
    }
}

#[derive(Clone, Debug, Default)]
/// Top-level configuration for [`OfflineSpeechDenoiser`].
pub struct OfflineSpeechDenoiserConfig {
    pub model: OfflineSpeechDenoiserModelConfig,
}

impl OfflineSpeechDenoiserConfig {
    fn to_sys(&self, cstrings: &mut Vec<CString>) -> sys::OfflineSpeechDenoiserConfig {
        sys::OfflineSpeechDenoiserConfig {
            model: self
                .model
                .to_sys(cstrings),
        }
    }
}

/// Offline speech denoiser.
pub struct OfflineSpeechDenoiser {
    ptr: *const sys::OfflineSpeechDenoiser,
}

// SAFETY: The sherpa-onnx C library is thread-safe for single-object usage.
unsafe impl Send for OfflineSpeechDenoiser {}
unsafe impl Sync for OfflineSpeechDenoiser {}

impl OfflineSpeechDenoiser {
    /// Create a denoiser from `config`.
    pub fn create(config: &OfflineSpeechDenoiserConfig) -> Option<Self> {
        let mut cstrings = Vec::new();
        let sys_config = config.to_sys(&mut cstrings);
        let ptr = unsafe { sys::SherpaOnnxCreateOfflineSpeechDenoiser(&sys_config) };
        if ptr.is_null() {
            None
        } else {
            Some(Self { ptr })
        }
    }

    /// Denoise one chunk or a complete waveform.
    pub fn run(&self, samples: &[f32], sample_rate: i32) -> DenoisedAudio {
        let samples_ptr = if samples.is_empty() {
            ptr::null()
        } else {
            samples.as_ptr()
        };
        let ptr = unsafe {
            sys::SherpaOnnxOfflineSpeechDenoiserRun(
                self.ptr,
                samples_ptr,
                samples.len() as i32,
                sample_rate,
            )
        };
        DenoisedAudio::from_ptr(ptr)
    }

    /// Return the model sample rate expected by this denoiser.
    pub fn sample_rate(&self) -> i32 {
        unsafe { sys::SherpaOnnxOfflineSpeechDenoiserGetSampleRate(self.ptr) }
    }
}

impl Drop for OfflineSpeechDenoiser {
    fn drop(&mut self) {
        unsafe { sys::SherpaOnnxDestroyOfflineSpeechDenoiser(self.ptr) }
    }
}
