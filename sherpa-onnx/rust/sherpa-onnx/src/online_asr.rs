use serde::Deserialize;
use std::ffi::{CStr, CString};
use std::os::raw::c_char;
use std::ptr;

use sherpa_onnx_sys as sys;

#[derive(Clone, Debug)]
pub struct OnlineTransducerModelConfig {
    pub encoder: Option<String>,
    pub decoder: Option<String>,
    pub joiner: Option<String>,
}

impl Default for OnlineTransducerModelConfig {
    fn default() -> Self {
        Self {
            encoder: None,
            decoder: None,
            joiner: None,
        }
    }
}

impl OnlineTransducerModelConfig {
    fn to_sys(&self, cstrings: &mut Vec<CString>) -> sys::OnlineTransducerModelConfig {
        sys::OnlineTransducerModelConfig {
            encoder: to_c_ptr(&self.encoder, cstrings),
            decoder: to_c_ptr(&self.decoder, cstrings),
            joiner: to_c_ptr(&self.joiner, cstrings),
        }
    }
}

#[derive(Clone, Debug)]
pub struct OnlineParaformerModelConfig {
    pub encoder: Option<String>,
    pub decoder: Option<String>,
}

impl Default for OnlineParaformerModelConfig {
    fn default() -> Self {
        Self {
            encoder: None,
            decoder: None,
        }
    }
}

impl OnlineParaformerModelConfig {
    fn to_sys(&self, cstrings: &mut Vec<CString>) -> sys::OnlineParaformerModelConfig {
        sys::OnlineParaformerModelConfig {
            encoder: to_c_ptr(&self.encoder, cstrings),
            decoder: to_c_ptr(&self.decoder, cstrings),
        }
    }
}

#[derive(Clone, Debug)]
pub struct OnlineZipformer2CtcModelConfig {
    pub model: Option<String>,
}

impl Default for OnlineZipformer2CtcModelConfig {
    fn default() -> Self {
        Self { model: None }
    }
}

impl OnlineZipformer2CtcModelConfig {
    fn to_sys(&self, cstrings: &mut Vec<CString>) -> sys::OnlineZipformer2CtcModelConfig {
        sys::OnlineZipformer2CtcModelConfig {
            model: to_c_ptr(&self.model, cstrings),
        }
    }
}

#[derive(Clone, Debug)]
pub struct OnlineNemoCtcModelConfig {
    pub model: Option<String>,
}

impl Default for OnlineNemoCtcModelConfig {
    fn default() -> Self {
        Self { model: None }
    }
}

impl OnlineNemoCtcModelConfig {
    fn to_sys(&self, cstrings: &mut Vec<CString>) -> sys::OnlineNemoCtcModelConfig {
        sys::OnlineNemoCtcModelConfig {
            model: to_c_ptr(&self.model, cstrings),
        }
    }
}

#[derive(Clone, Debug)]
pub struct OnlineToneCtcModelConfig {
    pub model: Option<String>,
}

impl Default for OnlineToneCtcModelConfig {
    fn default() -> Self {
        Self { model: None }
    }
}

impl OnlineToneCtcModelConfig {
    fn to_sys(&self, cstrings: &mut Vec<CString>) -> sys::OnlineToneCtcModelConfig {
        sys::OnlineToneCtcModelConfig {
            model: to_c_ptr(&self.model, cstrings),
        }
    }
}

#[derive(Clone, Debug)]
pub struct OnlineModelConfig {
    pub transducer: OnlineTransducerModelConfig,
    pub paraformer: OnlineParaformerModelConfig,
    pub zipformer2_ctc: OnlineZipformer2CtcModelConfig,
    pub nemo_ctc: OnlineNemoCtcModelConfig,
    pub t_one_ctc: OnlineToneCtcModelConfig,

    pub tokens: Option<String>,
    pub num_threads: i32,
    pub provider: Option<String>,
    pub debug: bool,

    pub model_type: Option<String>,
    pub modeling_unit: Option<String>, // cjkchar | bpe | cjkchar+bpe
    pub bpe_vocab: Option<String>,

    /// Optional in-memory tokens
    pub tokens_buf: Option<Vec<u8>>,
}

impl Default for OnlineModelConfig {
    fn default() -> Self {
        Self {
            transducer: Default::default(),
            paraformer: Default::default(),
            zipformer2_ctc: Default::default(),
            nemo_ctc: Default::default(),
            t_one_ctc: Default::default(),

            tokens: None,
            num_threads: 1,
            provider: Some("cpu".to_string()),
            debug: false,

            model_type: None,
            modeling_unit: None,
            bpe_vocab: None,
            tokens_buf: None,
        }
    }
}

impl OnlineModelConfig {
    fn to_sys(&self, cstrings: &mut Vec<CString>) -> sys::OnlineModelConfig {
        sys::OnlineModelConfig {
            transducer: self
                .transducer
                .to_sys(cstrings),
            paraformer: self
                .paraformer
                .to_sys(cstrings),
            zipformer2_ctc: self
                .zipformer2_ctc
                .to_sys(cstrings),
            nemo_ctc: self
                .nemo_ctc
                .to_sys(cstrings),
            t_one_ctc: self
                .t_one_ctc
                .to_sys(cstrings),

            tokens: to_c_ptr(&self.tokens, cstrings),
            num_threads: self.num_threads,
            provider: to_c_ptr(&self.provider, cstrings),
            debug: self.debug as i32,

            model_type: to_c_ptr(&self.model_type, cstrings),
            modeling_unit: to_c_ptr(&self.modeling_unit, cstrings),
            bpe_vocab: to_c_ptr(&self.bpe_vocab, cstrings),

            tokens_buf: self
                .tokens_buf
                .as_ref()
                .map_or(ptr::null(), |buf| buf.as_ptr() as *const _),
            tokens_buf_size: self
                .tokens_buf
                .as_ref()
                .map_or(0, |buf| buf.len() as i32),
        }
    }
}

#[derive(Clone, Debug)]
pub struct OnlineCtcFstDecoderConfig {
    pub graph: Option<String>,
    pub max_active: i32,
}

impl Default for OnlineCtcFstDecoderConfig {
    fn default() -> Self {
        Self {
            graph: None,
            max_active: 0,
        }
    }
}

impl OnlineCtcFstDecoderConfig {
    /// Convert to sys struct using `to_c_ptr()`
    pub(crate) fn to_sys(&self, cstrings: &mut Vec<CString>) -> sys::OnlineCtcFstDecoderConfig {
        sys::OnlineCtcFstDecoderConfig {
            graph: to_c_ptr(&self.graph, cstrings),
            max_active: self.max_active,
        }
    }
}

#[derive(Clone, Debug)]
pub struct HomophoneReplacerConfig {
    pub dict_dir: Option<String>, // currently unused
    pub lexicon: Option<String>,
    pub rule_fsts: Option<String>,
}

impl Default for HomophoneReplacerConfig {
    fn default() -> Self {
        Self {
            dict_dir: None,
            lexicon: None,
            rule_fsts: None,
        }
    }
}

impl HomophoneReplacerConfig {
    pub(crate) fn to_sys(&self, cstrings: &mut Vec<CString>) -> sys::HomophoneReplacerConfig {
        sys::HomophoneReplacerConfig {
            dict_dir: to_c_ptr(&self.dict_dir, cstrings),
            lexicon: to_c_ptr(&self.lexicon, cstrings),
            rule_fsts: to_c_ptr(&self.rule_fsts, cstrings),
        }
    }
}

#[derive(Clone, Debug)]
pub struct OnlineRecognizerConfig {
    pub feat_config: sys::FeatureConfig,
    pub model_config: OnlineModelConfig,

    /// Decoding method: greedy_search | modified_beam_search
    pub decoding_method: Option<String>,

    /// Used only when decoding_method is modified_beam_search
    pub max_active_paths: i32,

    /// Endpoint detection
    pub enable_endpoint: bool,

    pub rule1_min_trailing_silence: f32,
    pub rule2_min_trailing_silence: f32,
    pub rule3_min_utterance_length: f32,

    pub hotwords_file: Option<String>,
    pub hotwords_score: f32,

    pub ctc_fst_decoder_config: OnlineCtcFstDecoderConfig,

    pub rule_fsts: Option<String>,
    pub rule_fars: Option<String>,

    pub blank_penalty: f32,

    pub hotwords_buf: Option<Vec<u8>>,

    pub hr: HomophoneReplacerConfig,
}

impl Default for OnlineRecognizerConfig {
    fn default() -> Self {
        Self {
            feat_config: sys::FeatureConfig {
                sample_rate: 16000,
                feature_dim: 80,
            },
            model_config: Default::default(),
            decoding_method: None,
            max_active_paths: 0,
            enable_endpoint: false,
            rule1_min_trailing_silence: 0.0,
            rule2_min_trailing_silence: 0.0,
            rule3_min_utterance_length: 0.0,
            hotwords_file: None,
            hotwords_score: 0.0,
            ctc_fst_decoder_config: Default::default(),
            rule_fsts: None,
            rule_fars: None,
            blank_penalty: 0.0,
            hotwords_buf: None,
            hr: Default::default(),
        }
    }
}

impl OnlineRecognizerConfig {
    /// Convert to sys struct for FFI call
    pub(crate) fn to_sys(&self, cstrings: &mut Vec<CString>) -> sys::OnlineRecognizerConfig {
        sys::OnlineRecognizerConfig {
            feat_config: self.feat_config,
            model_config: self
                .model_config
                .to_sys(cstrings),
            decoding_method: to_c_ptr(&self.decoding_method, cstrings),
            max_active_paths: self.max_active_paths,
            enable_endpoint: self.enable_endpoint as i32,
            rule1_min_trailing_silence: self.rule1_min_trailing_silence,
            rule2_min_trailing_silence: self.rule2_min_trailing_silence,
            rule3_min_utterance_length: self.rule3_min_utterance_length,
            hotwords_file: to_c_ptr(&self.hotwords_file, cstrings),
            hotwords_score: self.hotwords_score,
            ctc_fst_decoder_config: self
                .ctc_fst_decoder_config
                .to_sys(cstrings),
            rule_fsts: to_c_ptr(&self.rule_fsts, cstrings),
            rule_fars: to_c_ptr(&self.rule_fars, cstrings),
            blank_penalty: self.blank_penalty,
            hotwords_buf: self
                .hotwords_buf
                .as_ref()
                .map_or(ptr::null(), |buf| buf.as_ptr() as *const _),
            hotwords_buf_size: self
                .hotwords_buf
                .as_ref()
                .map_or(0, |buf| buf.len() as i32),
            hr: self
                .hr
                .to_sys(cstrings),
        }
    }
}

pub struct OnlineRecognizer {
    ptr: *const sys::OnlineRecognizer,
}

impl OnlineRecognizer {
    pub fn create(config: &OnlineRecognizerConfig) -> Option<Self> {
        let mut cstrings = Vec::new();

        let sys_config = config.to_sys(&mut cstrings);

        let ptr = unsafe { sys::SherpaOnnxCreateOnlineRecognizer(&sys_config) };

        if ptr.is_null() {
            None
        } else {
            Some(Self { ptr })
        }
    }

    pub fn create_stream(&self) -> OnlineStream {
        let ptr = unsafe { sys::SherpaOnnxCreateOnlineStream(self.ptr) };
        OnlineStream { ptr }
    }

    pub fn create_stream_with_hotwords(&self, hotwords: &str) -> OnlineStream {
        let c = CString::new(hotwords).unwrap();
        let ptr = unsafe { sys::SherpaOnnxCreateOnlineStreamWithHotwords(self.ptr, c.as_ptr()) };
        OnlineStream { ptr }
    }

    pub fn decode(&self, stream: &OnlineStream) {
        unsafe { sys::SherpaOnnxDecodeOnlineStream(self.ptr, stream.ptr) }
    }

    pub fn decode_multiple_streams(&self, streams: &[&OnlineStream]) {
        let ptrs: Vec<*const sys::OnlineStream> = streams
            .iter()
            .map(|s| s.ptr)
            .collect();
        unsafe {
            sys::SherpaOnnxDecodeMultipleOnlineStreams(self.ptr, ptrs.as_ptr(), ptrs.len() as i32)
        }
    }

    pub fn reset(&self, stream: &OnlineStream) {
        unsafe { sys::SherpaOnnxOnlineStreamReset(self.ptr, stream.ptr) }
    }

    pub fn is_endpoint(&self, stream: &OnlineStream) -> bool {
        unsafe { sys::SherpaOnnxOnlineStreamIsEndpoint(self.ptr, stream.ptr) != 0 }
    }

    pub fn is_ready(&self, stream: &OnlineStream) -> bool {
        unsafe { sys::SherpaOnnxIsOnlineStreamReady(self.ptr, stream.ptr) != 0 }
    }

    pub fn get_result(&self, stream: &OnlineStream) -> Option<RecognizerResult> {
        unsafe {
            let cstr = sys::SherpaOnnxGetOnlineStreamResultAsJson(self.ptr, stream.ptr);
            if cstr.is_null() {
                return None;
            }
            let s = CStr::from_ptr(cstr)
                .to_string_lossy()
                .into_owned();
            sys::SherpaOnnxDestroyOnlineStreamResultJson(cstr);
            serde_json::from_str(&s).ok()
        }
    }
}

#[derive(Clone, Debug, Deserialize)]
pub struct RecognizerResult {
    pub text: String,
    pub tokens: Vec<String>,
    pub timestamps: Option<Vec<f32>>,
    pub segment: Option<i32>,
    pub start_time: Option<f32>,
    pub is_final: bool,
}

impl Drop for OnlineRecognizer {
    fn drop(&mut self) {
        unsafe {
            sys::SherpaOnnxDestroyOnlineRecognizer(self.ptr);
        }
    }
}

pub struct OnlineStream {
    ptr: *const sys::OnlineStream,
}

impl OnlineStream {
    pub fn accept_waveform(&self, sample_rate: i32, samples: &[f32]) {
        unsafe {
            sys::SherpaOnnxOnlineStreamAcceptWaveform(
                self.ptr,
                sample_rate,
                samples.as_ptr(),
                samples.len() as i32,
            )
        }
    }

    pub fn input_finished(&self) {
        unsafe { sys::SherpaOnnxOnlineStreamInputFinished(self.ptr) }
    }
}

impl Drop for OnlineStream {
    fn drop(&mut self) {
        unsafe { sys::SherpaOnnxDestroyOnlineStream(self.ptr) }
    }
}

fn to_c_ptr(opt: &Option<String>, storage: &mut Vec<CString>) -> *const c_char {
    if let Some(s) = opt {
        let c = CString::new(s.as_str()).unwrap();
        let ptr = c.as_ptr();
        storage.push(c);
        ptr
    } else {
        ptr::null()
    }
}
