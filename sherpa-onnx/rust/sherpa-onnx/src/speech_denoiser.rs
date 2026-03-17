use crate::utils::to_c_ptr;
use sherpa_onnx_sys as sys;
use std::ffi::CString;
use std::slice;

#[derive(Clone, Debug, Default)]
pub struct OfflineSpeechDenoiserGtcrnModelConfig {
    pub model: Option<String>,
}

impl OfflineSpeechDenoiserGtcrnModelConfig {
    pub(crate) fn to_sys(
        &self,
        cstrings: &mut Vec<CString>,
    ) -> sys::OfflineSpeechDenoiserGtcrnModelConfig {
        sys::OfflineSpeechDenoiserGtcrnModelConfig {
            model: to_c_ptr(&self.model, cstrings),
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct OfflineSpeechDenoiserDpdfNetModelConfig {
    pub model: Option<String>,
}

impl OfflineSpeechDenoiserDpdfNetModelConfig {
    pub(crate) fn to_sys(
        &self,
        cstrings: &mut Vec<CString>,
    ) -> sys::OfflineSpeechDenoiserDpdfNetModelConfig {
        sys::OfflineSpeechDenoiserDpdfNetModelConfig {
            model: to_c_ptr(&self.model, cstrings),
        }
    }
}

#[derive(Clone, Debug)]
pub struct OfflineSpeechDenoiserModelConfig {
    pub gtcrn: OfflineSpeechDenoiserGtcrnModelConfig,
    pub dpdfnet: OfflineSpeechDenoiserDpdfNetModelConfig,
    pub num_threads: i32,
    pub debug: bool,
    pub provider: Option<String>,
}

impl Default for OfflineSpeechDenoiserModelConfig {
    fn default() -> Self {
        Self {
            gtcrn: Default::default(),
            dpdfnet: Default::default(),
            num_threads: 1,
            debug: false,
            provider: Some("cpu".to_string()),
        }
    }
}

impl OfflineSpeechDenoiserModelConfig {
    pub(crate) fn to_sys(&self, cstrings: &mut Vec<CString>) -> sys::OfflineSpeechDenoiserModelConfig {
        sys::OfflineSpeechDenoiserModelConfig {
            gtcrn: self.gtcrn.to_sys(cstrings),
            num_threads: self.num_threads,
            debug: self.debug as i32,
            provider: to_c_ptr(&self.provider, cstrings),
            dpdfnet: self.dpdfnet.to_sys(cstrings),
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct DenoisedAudio {
    pub samples: Vec<f32>,
    pub sample_rate: i32,
}

impl DenoisedAudio {
    pub(crate) fn from_ptr(ptr: *const sys::DenoisedAudio) -> Self {
        if ptr.is_null() {
            return Self::default();
        }

        unsafe {
            let n = (*ptr).n.max(0) as usize;
            let samples = if (*ptr).samples.is_null() || n == 0 {
                vec![]
            } else {
                slice::from_raw_parts((*ptr).samples, n).to_vec()
            };
            let sample_rate = (*ptr).sample_rate;
            sys::SherpaOnnxDestroyDenoisedAudio(ptr);
            Self {
                samples,
                sample_rate,
            }
        }
    }
}
