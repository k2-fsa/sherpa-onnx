//! Offline audio tagging.
//!
//! This API classifies complete audio clips and returns the most likely events.
//! See:
//!
//! - `rust-api-examples/examples/audio_tagging_zipformer.rs`
//! - `rust-api-examples/examples/audio_tagging_ced.rs`

use crate::utils::to_c_ptr;
use sherpa_onnx_sys as sys;
use std::ffi::{CStr, CString};

#[derive(Clone, Debug, Default)]
/// Zipformer audio tagging model path.
pub struct OfflineZipformerAudioTaggingModelConfig {
    pub model: Option<String>,
}

impl OfflineZipformerAudioTaggingModelConfig {
    fn to_sys(&self, cstrings: &mut Vec<CString>) -> sys::OfflineZipformerAudioTaggingModelConfig {
        sys::OfflineZipformerAudioTaggingModelConfig {
            model: to_c_ptr(&self.model, cstrings),
        }
    }
}

#[derive(Clone, Debug)]
/// Model-level configuration for audio tagging.
///
/// Configure either `zipformer` or `ced` for a concrete model package.
pub struct AudioTaggingModelConfig {
    pub zipformer: OfflineZipformerAudioTaggingModelConfig,
    pub ced: Option<String>,
    pub num_threads: i32,
    pub debug: bool,
    pub provider: Option<String>,
}

impl Default for AudioTaggingModelConfig {
    fn default() -> Self {
        Self {
            zipformer: Default::default(),
            ced: None,
            num_threads: 1,
            debug: false,
            provider: Some("cpu".to_string()),
        }
    }
}

impl AudioTaggingModelConfig {
    fn to_sys(&self, cstrings: &mut Vec<CString>) -> sys::AudioTaggingModelConfig {
        sys::AudioTaggingModelConfig {
            zipformer: self
                .zipformer
                .to_sys(cstrings),
            ced: to_c_ptr(&self.ced, cstrings),
            num_threads: self.num_threads,
            debug: self.debug as i32,
            provider: to_c_ptr(&self.provider, cstrings),
        }
    }
}

#[derive(Clone, Debug)]
/// Top-level configuration for [`AudioTagging`].
pub struct AudioTaggingConfig {
    pub model: AudioTaggingModelConfig,
    pub labels: Option<String>,
    pub top_k: i32,
}

impl Default for AudioTaggingConfig {
    fn default() -> Self {
        Self {
            model: Default::default(),
            labels: None,
            top_k: 5,
        }
    }
}

impl AudioTaggingConfig {
    fn to_sys(&self, cstrings: &mut Vec<CString>) -> sys::AudioTaggingConfig {
        sys::AudioTaggingConfig {
            model: self
                .model
                .to_sys(cstrings),
            labels: to_c_ptr(&self.labels, cstrings),
            top_k: self.top_k,
        }
    }
}

#[derive(Clone, Debug)]
/// One predicted audio event.
pub struct AudioEvent {
    pub name: String,
    pub index: i32,
    pub prob: f32,
}

/// Offline audio tagger.
pub struct AudioTagging {
    ptr: *const sys::AudioTagging,
}

unsafe impl Send for AudioTagging {}

impl AudioTagging {
    /// Create a tagger from `config`.
    pub fn create(config: &AudioTaggingConfig) -> Option<Self> {
        let mut cstrings = Vec::new();
        let sys_config = config.to_sys(&mut cstrings);
        let ptr = unsafe { sys::SherpaOnnxCreateAudioTagging(&sys_config) };
        if ptr.is_null() {
            None
        } else {
            Some(Self { ptr })
        }
    }

    /// Create a stream that accepts one complete clip.
    pub fn create_stream(&self) -> AudioTaggingOfflineStream {
        let ptr = unsafe { sys::SherpaOnnxAudioTaggingCreateOfflineStream(self.ptr) };
        AudioTaggingOfflineStream { ptr }
    }

    /// Compute the top `top_k` events for the provided stream.
    pub fn compute(&self, stream: &AudioTaggingOfflineStream, top_k: i32) -> Vec<AudioEvent> {
        unsafe {
            let p = sys::SherpaOnnxAudioTaggingCompute(self.ptr, stream.ptr, top_k);
            if p.is_null() {
                return Vec::new();
            }

            let mut ans = Vec::new();
            let mut cur = p;
            while !(*cur).is_null() {
                let event = &*(*cur);
                let name = if event
                    .name
                    .is_null()
                {
                    String::new()
                } else {
                    CStr::from_ptr(event.name)
                        .to_string_lossy()
                        .into_owned()
                };
                ans.push(AudioEvent {
                    name,
                    index: event.index,
                    prob: event.prob,
                });
                cur = cur.add(1);
            }

            sys::SherpaOnnxAudioTaggingFreeResults(p);
            ans
        }
    }
}

impl Drop for AudioTagging {
    fn drop(&mut self) {
        unsafe {
            if !self
                .ptr
                .is_null()
            {
                sys::SherpaOnnxDestroyAudioTagging(self.ptr);
            }
        }
    }
}

/// Input stream for offline audio tagging.
pub struct AudioTaggingOfflineStream {
    ptr: *const sys::OfflineStream,
}

impl AudioTaggingOfflineStream {
    /// Append waveform samples to the clip.
    pub fn accept_waveform(&self, sample_rate: i32, samples: &[f32]) {
        unsafe {
            sys::SherpaOnnxAcceptWaveformOffline(
                self.ptr,
                sample_rate,
                samples.as_ptr(),
                samples.len() as i32,
            )
        }
    }
}

impl Drop for AudioTaggingOfflineStream {
    fn drop(&mut self) {
        unsafe { sys::SherpaOnnxDestroyOfflineStream(self.ptr) }
    }
}
