//! Voice activity detection and buffering helpers.
//!
//! See `rust-api-examples/examples/silero_vad_remove_silence.rs` for a complete
//! example that removes non-speech segments from a WAV file.

use crate::utils::to_c_ptr;
use std::ffi::CString;
use std::slice;

use sherpa_onnx_sys as sys;

#[derive(Clone, Debug, Default)]
/// Silero VAD configuration.
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
/// Ten VAD configuration.
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
/// Top-level model configuration for [`VoiceActivityDetector`].
///
/// Configure exactly one model family for typical use.
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

/// Circular sample buffer used by some VAD workflows.
pub struct CircularBuffer {
    ptr: *const sys::CircularBuffer,
}

// SAFETY: The sherpa-onnx C library is thread-safe for single-object usage.
unsafe impl Send for CircularBuffer {}
unsafe impl Sync for CircularBuffer {}

impl CircularBuffer {
    /// Create a new buffer with capacity measured in samples.
    pub fn new(capacity: i32) -> Option<Self> {
        let ptr = unsafe { sys::SherpaOnnxCreateCircularBuffer(capacity) };
        if ptr.is_null() {
            None
        } else {
            Some(Self { ptr })
        }
    }

    /// Append samples to the tail of the buffer.
    pub fn push(&self, samples: &[f32]) {
        unsafe {
            sys::SherpaOnnxCircularBufferPush(self.ptr, samples.as_ptr(), samples.len() as i32)
        }
    }

    /// Copy `n` samples starting at `start_index`.
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

    /// Drop `n` samples from the head of the buffer.
    pub fn pop(&self, n: i32) {
        unsafe { sys::SherpaOnnxCircularBufferPop(self.ptr, n) }
    }

    /// Return the number of samples currently stored.
    pub fn size(&self) -> i32 {
        unsafe { sys::SherpaOnnxCircularBufferSize(self.ptr) }
    }

    /// Return the logical head position.
    pub fn head(&self) -> i32 {
        unsafe { sys::SherpaOnnxCircularBufferHead(self.ptr) }
    }

    /// Clear the buffer.
    pub fn reset(&self) {
        unsafe { sys::SherpaOnnxCircularBufferReset(self.ptr) }
    }
}

impl Drop for CircularBuffer {
    fn drop(&mut self) {
        unsafe { sys::SherpaOnnxDestroyCircularBuffer(self.ptr) }
    }
}

/// One detected speech segment.
pub struct SpeechSegment {
    ptr: *const sys::SpeechSegment,
}

// SAFETY: The sherpa-onnx C library is thread-safe for single-object usage.
unsafe impl Send for SpeechSegment {}
unsafe impl Sync for SpeechSegment {}

impl SpeechSegment {
    /// Start index, in samples, relative to the input seen so far.
    pub fn start(&self) -> i32 {
        unsafe { (*self.ptr).start }
    }

    /// Borrow the segment samples.
    pub fn samples(&self) -> &[f32] {
        unsafe { slice::from_raw_parts((*self.ptr).samples, (*self.ptr).n as usize) }
    }

    /// Return the number of samples in the segment.
    pub fn n(&self) -> i32 {
        unsafe { (*self.ptr).n }
    }
}

impl Drop for SpeechSegment {
    fn drop(&mut self) {
        unsafe { sys::SherpaOnnxDestroySpeechSegment(self.ptr) }
    }
}

/// Voice activity detector that emits speech segments.
pub struct VoiceActivityDetector {
    ptr: *const sys::VoiceActivityDetector,
}

// SAFETY: The sherpa-onnx C library is thread-safe for single-object usage.
unsafe impl Send for VoiceActivityDetector {}
unsafe impl Sync for VoiceActivityDetector {}

impl VoiceActivityDetector {
    /// Create a detector and an internal result buffer.
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

    /// Feed waveform samples to the detector.
    pub fn accept_waveform(&self, samples: &[f32]) {
        unsafe {
            sys::SherpaOnnxVoiceActivityDetectorAcceptWaveform(
                self.ptr,
                samples.as_ptr(),
                samples.len() as i32,
            )
        }
    }

    /// Return `true` if there are no queued speech segments.
    pub fn is_empty(&self) -> bool {
        unsafe { sys::SherpaOnnxVoiceActivityDetectorEmpty(self.ptr) != 0 }
    }

    /// Return `true` if speech is currently being detected.
    pub fn detected(&self) -> bool {
        unsafe { sys::SherpaOnnxVoiceActivityDetectorDetected(self.ptr) != 0 }
    }

    /// Drop the front speech segment, if any.
    pub fn pop(&self) {
        unsafe { sys::SherpaOnnxVoiceActivityDetectorPop(self.ptr) }
    }

    /// Remove all queued segments.
    pub fn clear(&self) {
        unsafe { sys::SherpaOnnxVoiceActivityDetectorClear(self.ptr) }
    }

    /// Borrow the front speech segment, if available.
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

    /// Reset the detector state.
    pub fn reset(&self) {
        unsafe { sys::SherpaOnnxVoiceActivityDetectorReset(self.ptr) }
    }

    /// Flush any buffered trailing speech into the output queue.
    pub fn flush(&self) {
        unsafe { sys::SherpaOnnxVoiceActivityDetectorFlush(self.ptr) }
    }
}

impl Drop for VoiceActivityDetector {
    fn drop(&mut self) {
        unsafe { sys::SherpaOnnxDestroyVoiceActivityDetector(self.ptr) }
    }
}
