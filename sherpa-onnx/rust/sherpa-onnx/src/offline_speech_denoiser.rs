use crate::speech_denoiser::{DenoisedAudio, OfflineSpeechDenoiserModelConfig};
use sherpa_onnx_sys as sys;
use std::ffi::CString;
use std::ptr;

#[derive(Clone, Debug, Default)]
pub struct OfflineSpeechDenoiserConfig {
    pub model: OfflineSpeechDenoiserModelConfig,
}

impl OfflineSpeechDenoiserConfig {
    fn to_sys(&self, cstrings: &mut Vec<CString>) -> sys::OfflineSpeechDenoiserConfig {
        sys::OfflineSpeechDenoiserConfig {
            model: self.model.to_sys(cstrings),
        }
    }
}

pub struct OfflineSpeechDenoiser {
    ptr: *const sys::OfflineSpeechDenoiser,
}

impl OfflineSpeechDenoiser {
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

    pub fn sample_rate(&self) -> i32 {
        unsafe { sys::SherpaOnnxOfflineSpeechDenoiserGetSampleRate(self.ptr) }
    }
}

impl Drop for OfflineSpeechDenoiser {
    fn drop(&mut self) {
        unsafe { sys::SherpaOnnxDestroyOfflineSpeechDenoiser(self.ptr) }
    }
}
