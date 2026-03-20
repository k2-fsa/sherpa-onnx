use crate::utils::to_c_ptr;
use sherpa_onnx_sys as sys;
use std::ffi::{CStr, CString};

#[derive(Clone, Debug)]
pub struct OfflinePunctuationModelConfig {
    pub ct_transformer: Option<String>,
    pub num_threads: i32,
    pub debug: bool,
    pub provider: Option<String>,
}

impl Default for OfflinePunctuationModelConfig {
    fn default() -> Self {
        Self {
            ct_transformer: None,
            num_threads: 1,
            debug: false,
            provider: Some("cpu".to_string()),
        }
    }
}

impl OfflinePunctuationModelConfig {
    fn to_sys(&self, cstrings: &mut Vec<CString>) -> sys::OfflinePunctuationModelConfig {
        sys::OfflinePunctuationModelConfig {
            ct_transformer: to_c_ptr(&self.ct_transformer, cstrings),
            num_threads: self.num_threads,
            debug: self.debug as i32,
            provider: to_c_ptr(&self.provider, cstrings),
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct OfflinePunctuationConfig {
    pub model: OfflinePunctuationModelConfig,
}

impl OfflinePunctuationConfig {
    fn to_sys(&self, cstrings: &mut Vec<CString>) -> sys::OfflinePunctuationConfig {
        sys::OfflinePunctuationConfig {
            model: self.model.to_sys(cstrings),
        }
    }
}

pub struct OfflinePunctuation {
    ptr: *const sys::OfflinePunctuation,
}

unsafe impl Send for OfflinePunctuation {}

impl OfflinePunctuation {
    pub fn create(config: &OfflinePunctuationConfig) -> Option<Self> {
        let mut cstrings = Vec::new();
        let sys_config = config.to_sys(&mut cstrings);
        let ptr = unsafe { sys::SherpaOnnxCreateOfflinePunctuation(&sys_config) };
        if ptr.is_null() {
            None
        } else {
            Some(Self { ptr })
        }
    }

    pub fn add_punctuation(&self, text: &str) -> Option<String> {
        let text = CString::new(text).ok()?;

        unsafe {
            let p = sys::SherpaOfflinePunctuationAddPunct(self.ptr, text.as_ptr());
            if p.is_null() {
                return None;
            }

            let ans = CStr::from_ptr(p).to_string_lossy().into_owned();
            sys::SherpaOfflinePunctuationFreeText(p);
            Some(ans)
        }
    }
}

impl Drop for OfflinePunctuation {
    fn drop(&mut self) {
        unsafe {
            if !self.ptr.is_null() {
                sys::SherpaOnnxDestroyOfflinePunctuation(self.ptr);
            }
        }
    }
}
