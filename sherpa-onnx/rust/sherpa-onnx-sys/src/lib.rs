//! Raw FFI bindings for [sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx).
//!
//! # Environment variables
//!
//! The build script (`build.rs`) recognises the following environment
//! variables, all of which are optional:
//!
//! | Variable | Purpose |
//! |---|---|
//! | `SHERPA_ONNX_LIB_DIR` | Path to a directory that already contains the |
//! | | pre-built native libraries (`.so` / `.dylib` / `.dll` / `.a`). |
//! | | When set, the build script skips downloading and uses these |
//! | | libraries directly. |
//! | `SHERPA_ONNX_ARCHIVE_DIR` | Path to a directory that contains |
//! | | pre-downloaded archive files (`.tar.bz2` or `.xcframework.zip`). |
//! | | When set, the build script copies the archive from this directory |
//! | | instead of downloading it from the internet. The archive must have |
//! | | the same filename that would normally be downloaded (e.g. |
//! | | `sherpa-onnx-v1.13.8-linux-x64-shared-lib.tar.bz2`). |

#![allow(non_camel_case_types)]
#![allow(non_snake_case)]
#![allow(non_upper_case_globals)]

use std::os::raw::c_char;

extern "C" {
    pub fn SherpaOnnxGetVersionStr() -> *const c_char;
    pub fn SherpaOnnxGetGitSha1() -> *const c_char;
    pub fn SherpaOnnxGetGitDate() -> *const c_char;
    pub fn SherpaOnnxGetOnnxruntimeVersionStr() -> *const c_char;
    pub fn SherpaOnnxFileExists(filename: *const c_char) -> i32;
}

pub mod audio_tagging;
pub mod kws;
pub mod offline_asr;
pub mod offline_punctuation;
pub mod offline_speaker_diarization;
pub mod online_asr;
pub mod online_punctuation;
pub mod resampler;
pub mod speaker_embedding;
pub mod speech_denoiser;
pub mod spoken_language_identification;
pub mod tts;
pub mod vad;
pub mod wave;

pub use audio_tagging::*;
pub use kws::*;
pub use offline_asr::*;
pub use offline_punctuation::*;
pub use offline_speaker_diarization::*;
pub use online_asr::*;
pub use online_punctuation::*;
pub use resampler::*;
pub use speaker_embedding::*;
pub use speech_denoiser::*;
pub use spoken_language_identification::*;
pub use tts::*;
pub use vad::*;
pub use wave::*;
