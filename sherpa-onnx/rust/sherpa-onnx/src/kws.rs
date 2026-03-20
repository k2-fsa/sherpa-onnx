use crate::online_asr::{OnlineModelConfig, OnlineStream};
use crate::utils::to_c_ptr;
use sherpa_onnx_sys as sys;
use std::ffi::{c_char, CStr, CString};
use std::slice;

fn c_ptr_to_string(ptr: *const c_char) -> String {
    if ptr.is_null() {
        String::new()
    } else {
        unsafe { CStr::from_ptr(ptr).to_string_lossy().into_owned() }
    }
}

#[derive(Clone, Debug)]
pub struct KeywordSpotterConfig {
    pub feat_config: sys::FeatureConfig,
    pub model_config: OnlineModelConfig,
    pub max_active_paths: i32,
    pub num_trailing_blanks: i32,
    pub keywords_score: f32,
    pub keywords_threshold: f32,
    pub keywords_file: Option<String>,
    pub keywords_buf: Option<String>,
}

impl Default for KeywordSpotterConfig {
    fn default() -> Self {
        Self {
            feat_config: sys::FeatureConfig {
                sample_rate: 16000,
                feature_dim: 80,
            },
            model_config: Default::default(),
            max_active_paths: 4,
            num_trailing_blanks: 1,
            keywords_score: 1.0,
            keywords_threshold: 0.25,
            keywords_file: None,
            keywords_buf: None,
        }
    }
}

impl KeywordSpotterConfig {
    fn to_sys(&self, cstrings: &mut Vec<CString>) -> sys::KeywordSpotterConfig {
        sys::KeywordSpotterConfig {
            feat_config: self.feat_config,
            model_config: self.model_config.to_sys(cstrings),
            max_active_paths: self.max_active_paths,
            num_trailing_blanks: self.num_trailing_blanks,
            keywords_score: self.keywords_score,
            keywords_threshold: self.keywords_threshold,
            keywords_file: to_c_ptr(&self.keywords_file, cstrings),
            keywords_buf: to_c_ptr(&self.keywords_buf, cstrings),
            keywords_buf_size: self.keywords_buf.as_ref().map_or(0, |s| s.len() as i32),
        }
    }
}

#[derive(Clone, Debug)]
pub struct KeywordResult {
    pub keyword: String,
    pub tokens: String,
    pub tokens_arr: Vec<String>,
    pub timestamps: Vec<f32>,
    pub start_time: f32,
    pub json: String,
}

pub struct KeywordSpotter {
    ptr: *const sys::KeywordSpotter,
}

unsafe impl Send for KeywordSpotter {}

impl KeywordSpotter {
    pub fn create(config: &KeywordSpotterConfig) -> Option<Self> {
        let mut cstrings = Vec::new();
        let sys_config = config.to_sys(&mut cstrings);
        let ptr = unsafe { sys::SherpaOnnxCreateKeywordSpotter(&sys_config) };
        if ptr.is_null() {
            None
        } else {
            Some(Self { ptr })
        }
    }

    pub fn create_stream(&self) -> OnlineStream {
        let ptr = unsafe { sys::SherpaOnnxCreateKeywordStream(self.ptr) };
        OnlineStream { ptr }
    }

    pub fn create_stream_with_keywords(&self, keywords: &str) -> OnlineStream {
        let keywords = CString::new(keywords).unwrap();
        let ptr = unsafe {
            sys::SherpaOnnxCreateKeywordStreamWithKeywords(self.ptr, keywords.as_ptr())
        };
        OnlineStream { ptr }
    }

    pub fn is_ready(&self, stream: &OnlineStream) -> bool {
        unsafe { sys::SherpaOnnxIsKeywordStreamReady(self.ptr, stream.ptr) != 0 }
    }

    pub fn decode(&self, stream: &OnlineStream) {
        unsafe { sys::SherpaOnnxDecodeKeywordStream(self.ptr, stream.ptr) }
    }

    pub fn decode_multiple_streams(&self, streams: &[&OnlineStream]) {
        let ptrs: Vec<*const sys::OnlineStream> = streams.iter().map(|s| s.ptr).collect();
        unsafe {
            sys::SherpaOnnxDecodeMultipleKeywordStreams(self.ptr, ptrs.as_ptr(), ptrs.len() as i32)
        }
    }

    pub fn reset(&self, stream: &OnlineStream) {
        unsafe { sys::SherpaOnnxResetKeywordStream(self.ptr, stream.ptr) }
    }

    pub fn get_result(&self, stream: &OnlineStream) -> Option<KeywordResult> {
        unsafe {
            let p = sys::SherpaOnnxGetKeywordResult(self.ptr, stream.ptr);
            if p.is_null() {
                return None;
            }

            let result = &*p;
            let tokens_arr = if result.tokens_arr.is_null() || result.count <= 0 {
                Vec::new()
            } else {
                slice::from_raw_parts(result.tokens_arr, result.count as usize)
                    .iter()
                    .map(|item| c_ptr_to_string(*item))
                    .collect()
            };

            let timestamps = if result.timestamps.is_null() || result.count <= 0 {
                Vec::new()
            } else {
                slice::from_raw_parts(result.timestamps, result.count as usize).to_vec()
            };

            let ans = KeywordResult {
                keyword: c_ptr_to_string(result.keyword),
                tokens: c_ptr_to_string(result.tokens),
                tokens_arr,
                timestamps,
                start_time: result.start_time,
                json: c_ptr_to_string(result.json),
            };

            sys::SherpaOnnxDestroyKeywordResult(p);
            Some(ans)
        }
    }

    pub fn get_result_as_json(&self, stream: &OnlineStream) -> Option<String> {
        unsafe {
            let p = sys::SherpaOnnxGetKeywordResultAsJson(self.ptr, stream.ptr);
            if p.is_null() {
                return None;
            }

            let ans = CStr::from_ptr(p).to_string_lossy().into_owned();
            sys::SherpaOnnxFreeKeywordResultJson(p);
            Some(ans)
        }
    }
}

impl Drop for KeywordSpotter {
    fn drop(&mut self) {
        unsafe {
            if !self.ptr.is_null() {
                sys::SherpaOnnxDestroyKeywordSpotter(self.ptr);
            }
        }
    }
}
