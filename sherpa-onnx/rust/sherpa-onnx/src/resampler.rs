//! Linear resampler for converting between sample rates.

use std::slice;

use sherpa_onnx_sys as sys;

/// A linear resampler that converts audio from one sample rate to another.
pub struct LinearResampler {
    ptr: *const sys::SherpaOnnxLinearResampler,
}

impl LinearResampler {
    /// Create a new resampler that converts from `samp_rate_in_hz` to
    /// `samp_rate_out_hz`.
    pub fn create(samp_rate_in_hz: i32, samp_rate_out_hz: i32) -> Option<Self> {
        let filter_cutoff_hz =
            0.99 * 0.5 * (samp_rate_in_hz as f32).min(samp_rate_out_hz as f32);
        let num_zeros = 6;

        let ptr = unsafe {
            sys::SherpaOnnxCreateLinearResampler(
                samp_rate_in_hz,
                samp_rate_out_hz,
                filter_cutoff_hz,
                num_zeros,
            )
        };

        if ptr.is_null() {
            None
        } else {
            Some(Self { ptr })
        }
    }

    /// Resample the given audio samples.
    ///
    /// Set `flush` to `true` on the last chunk to flush any internal buffer.
    pub fn resample(&self, samples: &[f32], flush: bool) -> Vec<f32> {
        unsafe {
            let out = sys::SherpaOnnxLinearResamplerResample(
                self.ptr,
                samples.as_ptr(),
                samples.len() as i32,
                flush as i32,
            );
            if out.is_null() {
                return vec![];
            }
            let n = (*out).n as usize;
            let result = slice::from_raw_parts((*out).samples, n).to_vec();
            sys::SherpaOnnxLinearResamplerResampleFree(out);
            result
        }
    }

    /// Return the input sample rate.
    pub fn input_sample_rate(&self) -> i32 {
        unsafe { sys::SherpaOnnxLinearResamplerResampleGetInputSampleRate(self.ptr) }
    }

    /// Return the output sample rate.
    pub fn output_sample_rate(&self) -> i32 {
        unsafe { sys::SherpaOnnxLinearResamplerResampleGetOutputSampleRate(self.ptr) }
    }

    /// Reset the internal state of the resampler.
    pub fn reset(&self) {
        unsafe { sys::SherpaOnnxLinearResamplerReset(self.ptr) }
    }
}

impl Drop for LinearResampler {
    fn drop(&mut self) {
        unsafe { sys::SherpaOnnxDestroyLinearResampler(self.ptr) }
    }
}
