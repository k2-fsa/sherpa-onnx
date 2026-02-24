#![allow(non_camel_case_types)]
#![allow(non_snake_case)]
#![allow(non_upper_case_globals)]

use std::os::raw::{c_char, c_int};

#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct SherpaOnnxWave {
    /// Samples normalized to [-1, 1]
    pub samples: *const f32,
    pub sample_rate: c_int,
    pub num_samples: c_int,
}

extern "C" {
    /// Read a WAV file. Returns NULL on error.
    pub fn SherpaOnnxReadWave(filename: *const c_char) -> *const SherpaOnnxWave;

    /// Free memory allocated by SherpaOnnxReadWave
    pub fn SherpaOnnxFreeWave(wave: *const SherpaOnnxWave);
}
