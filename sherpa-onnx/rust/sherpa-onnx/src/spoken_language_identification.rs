use crate::offline_asr::OfflineStream;
use crate::utils::to_c_ptr;
use sherpa_onnx_sys as sys;
use std::ffi::{CStr, CString};

#[derive(Clone, Debug, Default)]
pub struct SpokenLanguageIdentificationWhisperConfig {
    pub encoder: Option<String>,
    pub decoder: Option<String>,
    pub tail_paddings: i32,
}

impl SpokenLanguageIdentificationWhisperConfig {
    fn to_sys(
        &self,
        cstrings: &mut Vec<CString>,
    ) -> sys::SpokenLanguageIdentificationWhisperConfig {
        sys::SpokenLanguageIdentificationWhisperConfig {
            encoder: to_c_ptr(&self.encoder, cstrings),
            decoder: to_c_ptr(&self.decoder, cstrings),
            tail_paddings: self.tail_paddings,
        }
    }
}

#[derive(Clone, Debug)]
pub struct SpokenLanguageIdentificationConfig {
    pub whisper: SpokenLanguageIdentificationWhisperConfig,
    pub num_threads: i32,
    pub debug: bool,
    pub provider: Option<String>,
}

impl Default for SpokenLanguageIdentificationConfig {
    fn default() -> Self {
        Self {
            whisper: Default::default(),
            num_threads: 1,
            debug: false,
            provider: Some("cpu".to_string()),
        }
    }
}

impl SpokenLanguageIdentificationConfig {
    fn to_sys(&self, cstrings: &mut Vec<CString>) -> sys::SpokenLanguageIdentificationConfig {
        sys::SpokenLanguageIdentificationConfig {
            whisper: self.whisper.to_sys(cstrings),
            num_threads: self.num_threads,
            debug: self.debug as i32,
            provider: to_c_ptr(&self.provider, cstrings),
        }
    }
}

#[derive(Clone, Debug)]
pub struct SpokenLanguageIdentificationResult {
    pub lang: String,
}

pub struct SpokenLanguageIdentification {
    ptr: *const sys::SpokenLanguageIdentification,
}

unsafe impl Send for SpokenLanguageIdentification {}

impl SpokenLanguageIdentification {
    pub fn create(config: &SpokenLanguageIdentificationConfig) -> Option<Self> {
        let mut cstrings = Vec::new();
        let sys_config = config.to_sys(&mut cstrings);
        let ptr = unsafe { sys::SherpaOnnxCreateSpokenLanguageIdentification(&sys_config) };
        if ptr.is_null() {
            None
        } else {
            Some(Self { ptr })
        }
    }

    pub fn create_stream(&self) -> OfflineStream {
        let ptr = unsafe { sys::SherpaOnnxSpokenLanguageIdentificationCreateOfflineStream(self.ptr) };
        OfflineStream { ptr }
    }

    pub fn compute(&self, stream: &OfflineStream) -> Option<SpokenLanguageIdentificationResult> {
        unsafe {
            let p = sys::SherpaOnnxSpokenLanguageIdentificationCompute(self.ptr, stream.ptr);
            if p.is_null() {
                return None;
            }

            let ans = SpokenLanguageIdentificationResult {
                lang: if (*p).lang.is_null() {
                    String::new()
                } else {
                    CStr::from_ptr((*p).lang).to_string_lossy().into_owned()
                },
            };

            sys::SherpaOnnxDestroySpokenLanguageIdentificationResult(p);
            Some(ans)
        }
    }
}

impl Drop for SpokenLanguageIdentification {
    fn drop(&mut self) {
        unsafe {
            if !self.ptr.is_null() {
                sys::SherpaOnnxDestroySpokenLanguageIdentification(self.ptr);
            }
        }
    }
}
