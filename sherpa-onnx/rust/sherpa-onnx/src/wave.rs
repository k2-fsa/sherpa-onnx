use std::ffi::CString;
use std::slice;

use sherpa_onnx_sys as sys;

#[derive(Debug)]
pub struct Wave {
    inner: *const sys::SherpaOnnxWave,
}

impl Wave {
    /// Read a WAV file using SherpaOnnx C API.
    pub fn read(filename: &str) -> Option<Self> {
        let c_filename = CString::new(filename).unwrap();
        let wave_ptr = unsafe { sys::SherpaOnnxReadWave(c_filename.as_ptr()) };
        if wave_ptr.is_null() {
            None
        } else {
            Some(Self { inner: wave_ptr })
        }
    }

    /// Get sample rate
    pub fn sample_rate(&self) -> i32 {
        unsafe { (*self.inner).sample_rate }
    }

    /// Get number of samples
    pub fn num_samples(&self) -> i32 {
        unsafe { (*self.inner).num_samples }
    }

    /// Get a slice of normalized samples
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
