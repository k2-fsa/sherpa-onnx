use crate::speech_denoiser::{DenoisedAudio, OfflineSpeechDenoiserModelConfig};
use sherpa_onnx_sys as sys;
use std::ffi::CString;
use std::ptr;

#[derive(Clone, Debug, Default)]
pub struct OnlineSpeechDenoiserConfig {
    pub model: OfflineSpeechDenoiserModelConfig,
}

impl OnlineSpeechDenoiserConfig {
    fn to_sys(&self, cstrings: &mut Vec<CString>) -> sys::OnlineSpeechDenoiserConfig {
        sys::OnlineSpeechDenoiserConfig {
            model: self.model.to_sys(cstrings),
        }
    }
}

pub struct OnlineSpeechDenoiser {
    ptr: *const sys::OnlineSpeechDenoiser,
}

impl OnlineSpeechDenoiser {
    pub fn create(config: &OnlineSpeechDenoiserConfig) -> Option<Self> {
        let mut cstrings = Vec::new();
        let sys_config = config.to_sys(&mut cstrings);
        let ptr = unsafe { sys::SherpaOnnxCreateOnlineSpeechDenoiser(&sys_config) };
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
            sys::SherpaOnnxOnlineSpeechDenoiserRun(
                self.ptr,
                samples_ptr,
                samples.len() as i32,
                sample_rate,
            )
        };
        DenoisedAudio::from_ptr(ptr)
    }

    pub fn flush(&self) -> DenoisedAudio {
        let ptr = unsafe { sys::SherpaOnnxOnlineSpeechDenoiserFlush(self.ptr) };
        DenoisedAudio::from_ptr(ptr)
    }

    pub fn reset(&self) {
        unsafe { sys::SherpaOnnxOnlineSpeechDenoiserReset(self.ptr) }
    }

    pub fn sample_rate(&self) -> i32 {
        unsafe { sys::SherpaOnnxOnlineSpeechDenoiserGetSampleRate(self.ptr) }
    }

    pub fn frame_shift_in_samples(&self) -> i32 {
        unsafe { sys::SherpaOnnxOnlineSpeechDenoiserGetFrameShiftInSamples(self.ptr) }
    }
}

impl Drop for OnlineSpeechDenoiser {
    fn drop(&mut self) {
        unsafe { sys::SherpaOnnxDestroyOnlineSpeechDenoiser(self.ptr) }
    }
}
