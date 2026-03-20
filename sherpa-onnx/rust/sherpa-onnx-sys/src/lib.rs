#![allow(non_camel_case_types)]
#![allow(non_snake_case)]
#![allow(non_upper_case_globals)]

use std::os::raw::c_char;

extern "C" {
    pub fn SherpaOnnxGetVersionStr() -> *const c_char;
    pub fn SherpaOnnxGetGitSha1() -> *const c_char;
    pub fn SherpaOnnxGetGitDate() -> *const c_char;
    pub fn SherpaOnnxFileExists(filename: *const c_char) -> i32;
}

pub mod audio_tagging;
pub mod offline_asr;
pub mod online_asr;
pub mod online_punctuation;
pub mod speech_denoiser;
pub mod tts;
pub mod vad;
pub mod wave;

pub use audio_tagging::*;
pub use offline_asr::*;
pub use online_asr::*;
pub use online_punctuation::*;
pub use speech_denoiser::*;
pub use tts::*;
pub use vad::*;
pub use wave::*;
