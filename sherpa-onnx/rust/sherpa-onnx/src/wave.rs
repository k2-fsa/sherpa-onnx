//! WAV file helpers used by the Rust wrappers and examples.

use std::ffi::CString;
use std::slice;

use sherpa_onnx_sys as sys;

#[derive(Debug)]
/// A WAV file loaded through sherpa-onnx.
///
/// Samples are exposed as normalized `f32` PCM values. Use [`Wave::read`] to
/// load a file and [`Wave::write`] or [`write()`] to save audio.
pub struct Wave {
    inner: *const sys::SherpaOnnxWave,
}

impl Wave {
    /// Read a mono WAV file from disk.
    ///
    /// Returns `None` if the file cannot be opened or decoded.
    pub fn read(filename: &str) -> Option<Self> {
        let c_filename = CString::new(filename).unwrap();
        let wave_ptr = unsafe { sys::SherpaOnnxReadWave(c_filename.as_ptr()) };
        if wave_ptr.is_null() {
            None
        } else {
            Some(Self { inner: wave_ptr })
        }
    }

    /// Write this waveform to a WAV file.
    pub fn write(&self, filename: &str) -> bool {
        let c_filename = CString::new(filename).unwrap();
        unsafe {
            sys::SherpaOnnxWriteWave(
                (*self.inner).samples,
                (*self.inner).num_samples,
                (*self.inner).sample_rate,
                c_filename.as_ptr(),
            ) == 1
        }
    }

    /// Return the sample rate in Hz.
    pub fn sample_rate(&self) -> i32 {
        unsafe { (*self.inner).sample_rate }
    }

    /// Return the number of samples in the waveform.
    pub fn num_samples(&self) -> i32 {
        unsafe { (*self.inner).num_samples }
    }

    /// Return the normalized PCM samples.
    pub fn samples(&self) -> &[f32] {
        unsafe {
            let ptr = (*self.inner).samples;
            let len = (*self.inner).num_samples as usize;

            if ptr.is_null() || len == 0 {
                &[]
            } else {
                slice::from_raw_parts(ptr, len)
            }
        }
    }
}

impl Drop for Wave {
    fn drop(&mut self) {
        unsafe {
            if !self
                .inner
                .is_null()
            {
                sys::SherpaOnnxFreeWave(self.inner);
            }
        }
    }
}

/// Write normalized PCM samples to a WAV file.
///
/// This is convenient when an API returns a plain `Vec<f32>` and you do not
/// need to build a [`Wave`] first.
pub fn write(filename: &str, samples: &[f32], sample_rate: i32) -> bool {
    let c_filename = CString::new(filename).unwrap();
    unsafe {
        sys::SherpaOnnxWriteWave(
            samples.as_ptr(),
            samples.len() as i32,
            sample_rate,
            c_filename.as_ptr(),
        ) == 1
    }
}
