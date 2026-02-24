// sherpa-onnx/src/vad.rs
use crate::utils::to_c_ptr;
use std::ffi::CString;
use std::slice;

use sherpa_onnx_sys as sys;

#[derive(Clone, Debug, Default)]
pub struct SileroVadModelConfig {
    pub model: Option<String>,
    pub threshold: f32,
    pub min_silence_duration: f32,
    pub min_speech_duration: f32,
    pub window_size: i32,
    pub max_speech_duration: f32,
}

impl SileroVadModelConfig {
    fn to_sys(&self, cstrings: &mut Vec<CString>) -> sys::SileroVadModelConfig {
        sys::SileroVadModelConfig {
            model: to_c_ptr(&self.model, cstrings),
            threshold: self.threshold,
            min_silence_duration: self.min_silence_duration,
            min_speech_duration: self.min_speech_duration,
            window_size: self.window_size,
            max_speech_duration: self.max_speech_duration,
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct TenVadModelConfig {
    pub model: Option<String>,
    pub threshold: f32,
    pub min_silence_duration: f32,
    pub min_speech_duration: f32,
    pub window_size: i32,
    pub max_speech_duration: f32,
}

impl TenVadModelConfig {
    fn to_sys(&self, cstrings: &mut Vec<CString>) -> sys::TenVadModelConfig {
        sys::TenVadModelConfig {
            model: to_c_ptr(&self.model, cstrings),
            threshold: self.threshold,
            min_silence_duration: self.min_silence_duration,
            min_speech_duration: self.min_speech_duration,
            window_size: self.window_size,
            max_speech_duration: self.max_speech_duration,
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct VadModelConfig {
    pub silero_vad: SileroVadModelConfig,
    pub ten_vad: TenVadModelConfig,
    pub sample_rate: i32,
    pub num_threads: i32,
    pub provider: Option<String>,
    pub debug: bool,
}

impl VadModelConfig {
    fn to_sys(&self, cstrings: &mut Vec<CString>) -> sys::VadModelConfig {
        sys::VadModelConfig {
            silero_vad: self
                .silero_vad
                .to_sys(cstrings),
            ten_vad: self
                .ten_vad
                .to_sys(cstrings),
            sample_rate: self.sample_rate,
            num_threads: self.num_threads,
            provider: to_c_ptr(&self.provider, cstrings),
            debug: self.debug as i32,
        }
    }
}

pub struct CircularBuffer {
    ptr: *const sys::CircularBuffer,
}

impl CircularBuffer {
    pub fn new(capacity: i32) -> Option<Self> {
        let ptr = unsafe { sys::SherpaOnnxCreateCircularBuffer(capacity) };
        if ptr.is_null() {
            None
        } else {
            Some(Self { ptr })
        }
    }

    pub fn push(&self, samples: &[f32]) {
        unsafe {
            sys::SherpaOnnxCircularBufferPush(self.ptr, samples.as_ptr(), samples.len() as i32)
        }
    }

    pub fn get(&self, start_index: i32, n: i32) -> Vec<f32> {
        unsafe {
            let p = sys::SherpaOnnxCircularBufferGet(self.ptr, start_index, n);
            if p.is_null() {
                return vec![];
            }
            let slice = slice::from_raw_parts(p, n as usize);
            let result = slice.to_vec();
            sys::SherpaOnnxCircularBufferFree(p);
            result
        }
    }

    pub fn pop(&self, n: i32) {
        unsafe { sys::SherpaOnnxCircularBufferPop(self.ptr, n) }
    }

    pub fn size(&self) -> i32 {
        unsafe { sys::SherpaOnnxCircularBufferSize(self.ptr) }
    }

    pub fn head(&self) -> i32 {
        unsafe { sys::SherpaOnnxCircularBufferHead(self.ptr) }
    }

    pub fn reset(&self) {
        unsafe { sys::SherpaOnnxCircularBufferReset(self.ptr) }
    }
}

impl Drop for CircularBuffer {
    fn drop(&mut self) {
        unsafe { sys::SherpaOnnxDestroyCircularBuffer(self.ptr) }
    }
}

pub struct SpeechSegment {
    ptr: *const sys::SpeechSegment,
}

impl SpeechSegment {
    pub fn start(&self) -> i32 {
        unsafe { (*self.ptr).start }
    }

    pub fn samples(&self) -> &[f32] {
        unsafe { slice::from_raw_parts((*self.ptr).samples, (*self.ptr).n as usize) }
    }

    pub fn n(&self) -> i32 {
        unsafe { (*self.ptr).n }
    }
}

impl Drop for SpeechSegment {
    fn drop(&mut self) {
        unsafe { sys::SherpaOnnxDestroySpeechSegment(self.ptr) }
    }
}

pub struct VoiceActivityDetector {
    ptr: *const sys::VoiceActivityDetector,
}

impl VoiceActivityDetector {
    pub fn create(config: &VadModelConfig, buffer_size_in_seconds: f32) -> Option<Self> {
        let mut cstrings = Vec::new();
        let sys_config = config.to_sys(&mut cstrings);

        let ptr = unsafe {
            sys::SherpaOnnxCreateVoiceActivityDetector(&sys_config, buffer_size_in_seconds)
        };

        if ptr.is_null() {
            None
        } else {
            Some(Self { ptr })
        }
    }

    pub fn accept_waveform(&self, samples: &[f32]) {
        unsafe {
            sys::SherpaOnnxVoiceActivityDetectorAcceptWaveform(
                self.ptr,
                samples.as_ptr(),
                samples.len() as i32,
            )
        }
    }

    pub fn is_empty(&self) -> bool {
        unsafe { sys::SherpaOnnxVoiceActivityDetectorEmpty(self.ptr) != 0 }
    }

    pub fn detected(&self) -> bool {
        unsafe { sys::SherpaOnnxVoiceActivityDetectorDetected(self.ptr) != 0 }
    }

    pub fn pop(&self) {
        unsafe { sys::SherpaOnnxVoiceActivityDetectorPop(self.ptr) }
    }

    pub fn clear(&self) {
        unsafe { sys::SherpaOnnxVoiceActivityDetectorClear(self.ptr) }
    }

    pub fn front(&self) -> Option<SpeechSegment> {
        if self.is_empty() {
            return None;
        }

        unsafe {
            let ptr = sys::SherpaOnnxVoiceActivityDetectorFront(self.ptr);
            if ptr.is_null() {
                None
            } else {
                Some(SpeechSegment { ptr })
            }
        }
    }

    pub fn reset(&self) {
        unsafe { sys::SherpaOnnxVoiceActivityDetectorReset(self.ptr) }
    }

    pub fn flush(&self) {
        unsafe { sys::SherpaOnnxVoiceActivityDetectorFlush(self.ptr) }
    }
}

impl Drop for VoiceActivityDetector {
    fn drop(&mut self) {
        unsafe { sys::SherpaOnnxDestroyVoiceActivityDetector(self.ptr) }
    }
}
