//! Offline speech recognition.
//!
//! The Rust wrapper exposes the same model families as the native C API. In
//! typical use, configure exactly one model family inside [`OfflineModelConfig`]
//! and then create an [`OfflineRecognizer`].
//!
//! Repository examples:
//!
//! - `rust-api-examples/examples/sense_voice.rs`
//! - `rust-api-examples/examples/nemo_parakeet.rs`
//! - `rust-api-examples/examples/moonshine_v2.rs`
//! - `rust-api-examples/examples/fire_red_asr_ctc.rs`
//!
//! ```no_run
//! use sherpa_onnx::{
//!     OfflineRecognizer, OfflineRecognizerConfig, OfflineSenseVoiceModelConfig, Wave,
//! };
//!
//! let wave = Wave::read("./test.wav").expect("read wave");
//! let mut config = OfflineRecognizerConfig::default();
//! config.model_config.sense_voice = OfflineSenseVoiceModelConfig {
//!     model: Some(
//!         "./sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17-int8/model.int8.onnx".into(),
//!     ),
//!     language: Some("auto".into()),
//!     use_itn: true,
//! };
//! config.model_config.tokens = Some(
//!     "./sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17-int8/tokens.txt".into(),
//! );
//!
//! let recognizer = OfflineRecognizer::create(&config).expect("create recognizer");
//! let stream = recognizer.create_stream();
//! stream.accept_waveform(wave.sample_rate(), wave.samples());
//! recognizer.decode(&stream);
//! println!("{}", stream.get_result().expect("result").text);
//! ```

use crate::utils::to_c_ptr;
use serde::Deserialize;
use sherpa_onnx_sys as sys;
use std::ffi::{CStr, CString};

#[derive(Clone, Debug, Default)]
/// Offline transducer model configuration.
///
/// This is used for transducer-style models such as the Parakeet example in
/// `rust-api-examples/examples/nemo_parakeet.rs`.
pub struct OfflineTransducerModelConfig {
    pub encoder: Option<String>,
    pub decoder: Option<String>,
    pub joiner: Option<String>,
}

impl OfflineTransducerModelConfig {
    fn to_sys(&self, cstrings: &mut Vec<CString>) -> sys::OfflineTransducerModelConfig {
        sys::OfflineTransducerModelConfig {
            encoder: to_c_ptr(&self.encoder, cstrings),
            decoder: to_c_ptr(&self.decoder, cstrings),
            joiner: to_c_ptr(&self.joiner, cstrings),
        }
    }
}

#[derive(Clone, Debug, Default)]
/// Offline Paraformer model configuration.
pub struct OfflineParaformerModelConfig {
    pub model: Option<String>,
}

impl OfflineParaformerModelConfig {
    fn to_sys(&self, cstrings: &mut Vec<CString>) -> sys::OfflineParaformerModelConfig {
        sys::OfflineParaformerModelConfig {
            model: to_c_ptr(&self.model, cstrings),
        }
    }
}

#[derive(Clone, Debug, Default)]
/// Offline NeMo CTC model configuration.
pub struct OfflineNemoEncDecCtcModelConfig {
    pub model: Option<String>,
}

impl OfflineNemoEncDecCtcModelConfig {
    fn to_sys(&self, cstrings: &mut Vec<CString>) -> sys::OfflineNemoEncDecCtcModelConfig {
        sys::OfflineNemoEncDecCtcModelConfig {
            model: to_c_ptr(&self.model, cstrings),
        }
    }
}

#[derive(Clone, Debug, Default)]
/// Offline Whisper model configuration.
pub struct OfflineWhisperModelConfig {
    pub encoder: Option<String>,
    pub decoder: Option<String>,
    pub language: Option<String>,
    pub task: Option<String>,
    pub tail_paddings: i32,
    pub enable_token_timestamps: bool,
    pub enable_segment_timestamps: bool,
}

impl OfflineWhisperModelConfig {
    fn to_sys(&self, cstrings: &mut Vec<CString>) -> sys::OfflineWhisperModelConfig {
        sys::OfflineWhisperModelConfig {
            encoder: to_c_ptr(&self.encoder, cstrings),
            decoder: to_c_ptr(&self.decoder, cstrings),
            language: to_c_ptr(&self.language, cstrings),
            task: to_c_ptr(&self.task, cstrings),
            tail_paddings: self.tail_paddings,
            enable_token_timestamps: self.enable_token_timestamps as i32,
            enable_segment_timestamps: self.enable_segment_timestamps as i32,
        }
    }
}

#[derive(Clone, Debug, Default)]
/// Offline Canary model configuration.
pub struct OfflineCanaryModelConfig {
    pub encoder: Option<String>,
    pub decoder: Option<String>,
    pub src_lang: Option<String>,
    pub tgt_lang: Option<String>,
    pub use_pnc: bool,
}

impl OfflineCanaryModelConfig {
    fn to_sys(&self, cstrings: &mut Vec<CString>) -> sys::OfflineCanaryModelConfig {
        sys::OfflineCanaryModelConfig {
            encoder: to_c_ptr(&self.encoder, cstrings),
            decoder: to_c_ptr(&self.decoder, cstrings),
            src_lang: to_c_ptr(&self.src_lang, cstrings),
            tgt_lang: to_c_ptr(&self.tgt_lang, cstrings),
            use_pnc: self.use_pnc as i32,
        }
    }
}

#[derive(Clone, Debug, Default)]
/// Offline FireRed ASR transducer configuration.
pub struct OfflineFireRedAsrModelConfig {
    pub encoder: Option<String>,
    pub decoder: Option<String>,
}

impl OfflineFireRedAsrModelConfig {
    fn to_sys(&self, cstrings: &mut Vec<CString>) -> sys::OfflineFireRedAsrModelConfig {
        sys::OfflineFireRedAsrModelConfig {
            encoder: to_c_ptr(&self.encoder, cstrings),
            decoder: to_c_ptr(&self.decoder, cstrings),
        }
    }
}

/// For Moonshine v1, you need 4 models:
///  - preprocessor, encoder, uncached_decoder, cached_decoder
///
/// For Moonshine v2, you need 2 models:
///  - encoder, merged_decoder
#[derive(Clone, Debug, Default)]
/// Offline Moonshine model configuration.
pub struct OfflineMoonshineModelConfig {
    pub preprocessor: Option<String>,
    pub encoder: Option<String>,
    pub uncached_decoder: Option<String>,
    pub cached_decoder: Option<String>,
    pub merged_decoder: Option<String>,
}

impl OfflineMoonshineModelConfig {
    fn to_sys(&self, cstrings: &mut Vec<CString>) -> sys::OfflineMoonshineModelConfig {
        sys::OfflineMoonshineModelConfig {
            preprocessor: to_c_ptr(&self.preprocessor, cstrings),
            encoder: to_c_ptr(&self.encoder, cstrings),
            uncached_decoder: to_c_ptr(&self.uncached_decoder, cstrings),
            cached_decoder: to_c_ptr(&self.cached_decoder, cstrings),
            merged_decoder: to_c_ptr(&self.merged_decoder, cstrings),
        }
    }
}

#[derive(Clone, Debug, Default)]
/// Offline TDNN model configuration.
pub struct OfflineTdnnModelConfig {
    pub model: Option<String>,
}

impl OfflineTdnnModelConfig {
    fn to_sys(&self, cstrings: &mut Vec<CString>) -> sys::OfflineTdnnModelConfig {
        sys::OfflineTdnnModelConfig {
            model: to_c_ptr(&self.model, cstrings),
        }
    }
}

#[derive(Clone, Debug)]
/// Optional external language model configuration for offline ASR.
pub struct OfflineLMConfig {
    pub model: Option<String>,
    pub scale: f32,
}
impl Default for OfflineLMConfig {
    fn default() -> Self {
        Self {
            model: None,
            scale: 1.0,
        }
    }
}
impl OfflineLMConfig {
    fn to_sys(&self, cstrings: &mut Vec<CString>) -> sys::OfflineLMConfig {
        sys::OfflineLMConfig {
            model: to_c_ptr(&self.model, cstrings),
            scale: self.scale,
        }
    }
}

#[derive(Clone, Debug, Default)]
/// Offline SenseVoice model configuration.
pub struct OfflineSenseVoiceModelConfig {
    pub model: Option<String>,
    pub language: Option<String>,
    pub use_itn: bool,
}

impl OfflineSenseVoiceModelConfig {
    fn to_sys(&self, cstrings: &mut Vec<CString>) -> sys::OfflineSenseVoiceModelConfig {
        sys::OfflineSenseVoiceModelConfig {
            model: to_c_ptr(&self.model, cstrings),
            language: to_c_ptr(&self.language, cstrings),
            use_itn: self.use_itn as i32,
        }
    }
}

#[derive(Clone, Debug, Default)]
/// Offline Dolphin model configuration.
pub struct OfflineDolphinModelConfig {
    pub model: Option<String>,
}

impl OfflineDolphinModelConfig {
    fn to_sys(&self, cstrings: &mut Vec<CString>) -> sys::OfflineDolphinModelConfig {
        sys::OfflineDolphinModelConfig {
            model: to_c_ptr(&self.model, cstrings),
        }
    }
}

#[derive(Clone, Debug, Default)]
/// Offline Zipformer CTC model configuration.
pub struct OfflineZipformerCtcModelConfig {
    pub model: Option<String>,
}

impl OfflineZipformerCtcModelConfig {
    fn to_sys(&self, cstrings: &mut Vec<CString>) -> sys::OfflineZipformerCtcModelConfig {
        sys::OfflineZipformerCtcModelConfig {
            model: to_c_ptr(&self.model, cstrings),
        }
    }
}

#[derive(Clone, Debug, Default)]
/// Offline WeNet CTC model configuration.
pub struct OfflineWenetCtcModelConfig {
    pub model: Option<String>,
}

impl OfflineWenetCtcModelConfig {
    fn to_sys(&self, cstrings: &mut Vec<CString>) -> sys::OfflineWenetCtcModelConfig {
        sys::OfflineWenetCtcModelConfig {
            model: to_c_ptr(&self.model, cstrings),
        }
    }
}

#[derive(Clone, Debug, Default)]
/// Offline omnilingual CTC model configuration.
pub struct OfflineOmnilingualAsrCtcModelConfig {
    pub model: Option<String>,
}

impl OfflineOmnilingualAsrCtcModelConfig {
    fn to_sys(&self, cstrings: &mut Vec<CString>) -> sys::OfflineOmnilingualAsrCtcModelConfig {
        sys::OfflineOmnilingualAsrCtcModelConfig {
            model: to_c_ptr(&self.model, cstrings),
        }
    }
}

#[derive(Clone, Debug, Default)]
/// Offline MedASR CTC model configuration.
pub struct OfflineMedAsrCtcModelConfig {
    pub model: Option<String>,
}

impl OfflineMedAsrCtcModelConfig {
    fn to_sys(&self, cstrings: &mut Vec<CString>) -> sys::OfflineMedAsrCtcModelConfig {
        sys::OfflineMedAsrCtcModelConfig {
            model: to_c_ptr(&self.model, cstrings),
        }
    }
}

#[derive(Clone, Debug, Default)]
/// Offline FireRed ASR CTC model configuration.
pub struct OfflineFireRedAsrCtcModelConfig {
    pub model: Option<String>,
}

impl OfflineFireRedAsrCtcModelConfig {
    fn to_sys(&self, cstrings: &mut Vec<CString>) -> sys::OfflineFireRedAsrCtcModelConfig {
        sys::OfflineFireRedAsrCtcModelConfig {
            model: to_c_ptr(&self.model, cstrings),
        }
    }
}

#[derive(Clone, Debug)]
/// Offline Qwen3 ASR model configuration.
pub struct OfflineQwen3ASRModelConfig {
    pub conv_frontend: Option<String>,
    pub encoder: Option<String>,
    pub decoder: Option<String>,
    pub tokenizer: Option<String>,
    pub max_total_len: i32,
    pub max_new_tokens: i32,
    pub temperature: f32,
    pub top_p: f32,
    pub seed: i32,
    pub hotwords: Option<String>,
}
impl Default for OfflineQwen3ASRModelConfig {
    fn default() -> Self {
        Self {
            conv_frontend: None,
            encoder: None,
            decoder: None,
            tokenizer: None,
            max_total_len: 512,
            max_new_tokens: 128,
            temperature: 1e-6,
            top_p: 0.8,
            seed: 42,
            hotwords: None,
        }
    }
}

#[derive(Clone, Debug, Default)]
/// Offline Cohere Transcribe model configuration.
pub struct OfflineCohereTranscribeModelConfig {
    pub encoder: Option<String>,
    pub decoder: Option<String>,
    pub language: Option<String>,
    pub use_punct: bool,
    pub use_itn: bool,
}

impl OfflineCohereTranscribeModelConfig {
    fn to_sys(&self, cstrings: &mut Vec<CString>) -> sys::OfflineCohereTranscribeModelConfig {
        sys::OfflineCohereTranscribeModelConfig {
            encoder: to_c_ptr(&self.encoder, cstrings),
            decoder: to_c_ptr(&self.decoder, cstrings),
            language: to_c_ptr(&self.language, cstrings),
            use_punct: self.use_punct as i32,
            use_itn: self.use_itn as i32,
        }
    }
}
impl OfflineQwen3ASRModelConfig {
    fn to_sys(&self, cstrings: &mut Vec<CString>) -> sys::OfflineQwen3ASRModelConfig {
        sys::OfflineQwen3ASRModelConfig {
            conv_frontend: to_c_ptr(&self.conv_frontend, cstrings),
            encoder: to_c_ptr(&self.encoder, cstrings),
            decoder: to_c_ptr(&self.decoder, cstrings),
            tokenizer: to_c_ptr(&self.tokenizer, cstrings),
            max_total_len: self.max_total_len,
            max_new_tokens: self.max_new_tokens,
            temperature: self.temperature,
            top_p: self.top_p,
            seed: self.seed,
            hotwords: to_c_ptr(&self.hotwords, cstrings),
        }
    }
}
#[derive(Clone, Debug)]
/// Offline FunASR Nano model configuration.
pub struct OfflineFunASRNanoModelConfig {
    pub encoder_adaptor: Option<String>,
    pub llm: Option<String>,
    pub embedding: Option<String>,
    pub tokenizer: Option<String>,
    pub system_prompt: Option<String>,
    pub user_prompt: Option<String>,
    pub max_new_tokens: i32,
    pub temperature: f32,
    pub top_p: f32,
    pub seed: i32,
    pub language: Option<String>,
    pub itn: i32,
    pub hotwords: Option<String>,
}
impl Default for OfflineFunASRNanoModelConfig {
    fn default() -> Self {
        Self {
            encoder_adaptor: None,
            llm: None,
            embedding: None,
            tokenizer: None,
            system_prompt: None,
            user_prompt: None,
            max_new_tokens: 0,
            temperature: 1.0,
            top_p: 1.0,
            seed: 0,
            language: None,
            itn: 0,
            hotwords: None,
        }
    }
}
impl OfflineFunASRNanoModelConfig {
    fn to_sys(&self, cstrings: &mut Vec<CString>) -> sys::OfflineFunASRNanoModelConfig {
        sys::OfflineFunASRNanoModelConfig {
            encoder_adaptor: to_c_ptr(&self.encoder_adaptor, cstrings),
            llm: to_c_ptr(&self.llm, cstrings),
            embedding: to_c_ptr(&self.embedding, cstrings),
            tokenizer: to_c_ptr(&self.tokenizer, cstrings),
            system_prompt: to_c_ptr(&self.system_prompt, cstrings),
            user_prompt: to_c_ptr(&self.user_prompt, cstrings),
            max_new_tokens: self.max_new_tokens,
            temperature: self.temperature,
            top_p: self.top_p,
            seed: self.seed,
            language: to_c_ptr(&self.language, cstrings),
            itn: self.itn,
            hotwords: to_c_ptr(&self.hotwords, cstrings),
        }
    }
}

#[derive(Clone, Debug, Default)]
/// Aggregate model configuration for offline recognition.
///
/// Configure exactly one model family for typical use. Shared options such as
/// `tokens`, `provider`, and `num_threads` live here as well.
pub struct OfflineModelConfig {
    pub transducer: OfflineTransducerModelConfig,
    pub paraformer: OfflineParaformerModelConfig,
    pub nemo_ctc: OfflineNemoEncDecCtcModelConfig,
    pub whisper: OfflineWhisperModelConfig,
    pub tdnn: OfflineTdnnModelConfig,
    pub sense_voice: OfflineSenseVoiceModelConfig,
    pub moonshine: OfflineMoonshineModelConfig,
    pub fire_red_asr: OfflineFireRedAsrModelConfig,
    pub dolphin: OfflineDolphinModelConfig,
    pub zipformer_ctc: OfflineZipformerCtcModelConfig,
    pub canary: OfflineCanaryModelConfig,
    pub wenet_ctc: OfflineWenetCtcModelConfig,
    pub omnilingual: OfflineOmnilingualAsrCtcModelConfig,
    pub medasr: OfflineMedAsrCtcModelConfig,
    pub funasr_nano: OfflineFunASRNanoModelConfig,
    pub fire_red_asr_ctc: OfflineFireRedAsrCtcModelConfig,
    pub qwen3_asr: OfflineQwen3ASRModelConfig,
    pub cohere_transcribe: OfflineCohereTranscribeModelConfig,

    pub tokens: Option<String>,
    pub num_threads: i32,
    pub debug: bool,
    pub provider: Option<String>,
    pub model_type: Option<String>,
    pub modeling_unit: Option<String>,
    pub bpe_vocab: Option<String>,
    pub telespeech_ctc: Option<String>,
}

impl OfflineModelConfig {
    fn to_sys(&self, cstrings: &mut Vec<CString>) -> sys::OfflineModelConfig {
        sys::OfflineModelConfig {
            transducer: self
                .transducer
                .to_sys(cstrings),
            paraformer: self
                .paraformer
                .to_sys(cstrings),
            nemo_ctc: self
                .nemo_ctc
                .to_sys(cstrings),
            whisper: self
                .whisper
                .to_sys(cstrings),
            tdnn: self
                .tdnn
                .to_sys(cstrings),
            sense_voice: self
                .sense_voice
                .to_sys(cstrings),
            canary: self
                .canary
                .to_sys(cstrings),
            fire_red_asr: self
                .fire_red_asr
                .to_sys(cstrings),
            dolphin: self
                .dolphin
                .to_sys(cstrings),
            moonshine: self
                .moonshine
                .to_sys(cstrings),
            zipformer_ctc: self
                .zipformer_ctc
                .to_sys(cstrings),
            wenet_ctc: self
                .wenet_ctc
                .to_sys(cstrings),
            omnilingual: self
                .omnilingual
                .to_sys(cstrings),
            medasr: self
                .medasr
                .to_sys(cstrings),
            funasr_nano: self
                .funasr_nano
                .to_sys(cstrings),
            fire_red_asr_ctc: self
                .fire_red_asr_ctc
                .to_sys(cstrings),
            qwen3_asr: self
                .qwen3_asr
                .to_sys(cstrings),
            cohere_transcribe: self
                .cohere_transcribe
                .to_sys(cstrings),

            tokens: to_c_ptr(&self.tokens, cstrings),
            num_threads: self.num_threads,
            debug: self.debug as i32,
            provider: to_c_ptr(&self.provider, cstrings),
            model_type: to_c_ptr(&self.model_type, cstrings),
            modeling_unit: to_c_ptr(&self.modeling_unit, cstrings),
            bpe_vocab: to_c_ptr(&self.bpe_vocab, cstrings),
            telespeech_ctc: to_c_ptr(&self.telespeech_ctc, cstrings),
        }
    }
}

#[derive(Clone, Debug)]
/// Top-level configuration for [`OfflineRecognizer`].
///
/// Use [`Default`] as a starting point, then fill the fields for the model you
/// want to run.
pub struct OfflineRecognizerConfig {
    pub feat_config: sys::FeatureConfig,
    pub model_config: OfflineModelConfig,
    pub lm_config: OfflineLMConfig,
    pub decoding_method: Option<String>,
    pub max_active_paths: i32,
    pub hotwords_file: Option<String>,
    pub hotwords_score: f32,
    pub rule_fsts: Option<String>,
    pub rule_fars: Option<String>,
    pub blank_penalty: f32,
    pub hr: super::online_asr::HomophoneReplacerConfig,
}

impl OfflineRecognizerConfig {
    fn to_sys(&self, cstrings: &mut Vec<CString>) -> sys::OfflineRecognizerConfig {
        sys::OfflineRecognizerConfig {
            feat_config: self.feat_config,
            model_config: self
                .model_config
                .to_sys(cstrings),
            lm_config: self
                .lm_config
                .to_sys(cstrings),
            decoding_method: to_c_ptr(&self.decoding_method, cstrings),
            max_active_paths: self.max_active_paths,
            hotwords_file: to_c_ptr(&self.hotwords_file, cstrings),
            hotwords_score: self.hotwords_score,
            rule_fsts: to_c_ptr(&self.rule_fsts, cstrings),
            rule_fars: to_c_ptr(&self.rule_fars, cstrings),
            blank_penalty: self.blank_penalty,
            hr: self
                .hr
                .to_sys(cstrings),
        }
    }
}

impl Default for OfflineRecognizerConfig {
    fn default() -> Self {
        Self {
            feat_config: sys::FeatureConfig {
                sample_rate: 16000,
                feature_dim: 80,
            },

            model_config: OfflineModelConfig::default(),
            lm_config: OfflineLMConfig::default(),
            decoding_method: None,
            max_active_paths: 4, // a reasonable default
            hotwords_file: None,
            hotwords_score: 0.0,
            rule_fsts: None,
            rule_fars: None,
            blank_penalty: 0.0,
            hr: super::online_asr::HomophoneReplacerConfig::default(),
        }
    }
}

#[derive(Clone, Debug, Deserialize)]
/// Recognition result returned by [`OfflineStream::get_result`].
pub struct OfflineRecognizerResult {
    pub text: String,
    pub tokens: Vec<String>,
    pub timestamps: Option<Vec<f32>>,
    pub durations: Option<Vec<f32>>,
}

/// Offline speech recognizer.
///
/// ```no_run
/// use sherpa_onnx::{
///     OfflineRecognizer, OfflineRecognizerConfig, OfflineTransducerModelConfig, Wave,
/// };
///
/// let wave = Wave::read("./test.wav").expect("read wave");
/// let mut config = OfflineRecognizerConfig::default();
/// config.model_config.transducer = OfflineTransducerModelConfig {
///     encoder: Some("./sherpa-onnx-nemo-parakeet-tdt-0.6b-v3-int8/encoder.int8.onnx".into()),
///     decoder: Some("./sherpa-onnx-nemo-parakeet-tdt-0.6b-v3-int8/decoder.int8.onnx".into()),
///     joiner: Some("./sherpa-onnx-nemo-parakeet-tdt-0.6b-v3-int8/joiner.int8.onnx".into()),
/// };
/// config.model_config.tokens =
///     Some("./sherpa-onnx-nemo-parakeet-tdt-0.6b-v3-int8/tokens.txt".into());
/// config.model_config.model_type = Some("nemo_transducer".into());
///
/// let recognizer = OfflineRecognizer::create(&config).expect("create recognizer");
/// let stream = recognizer.create_stream();
/// stream.accept_waveform(wave.sample_rate(), wave.samples());
/// recognizer.decode(&stream);
/// let result = stream.get_result().expect("result");
/// println!("{}", result.text);
/// ```
pub struct OfflineRecognizer {
    ptr: *const sys::OfflineRecognizer,
}

impl OfflineRecognizer {
    /// Create a recognizer from `config`.
    pub fn create(config: &OfflineRecognizerConfig) -> Option<Self> {
        let mut cstrings = Vec::new();
        let sys_config = config.to_sys(&mut cstrings);
        let ptr = unsafe { sys::SherpaOnnxCreateOfflineRecognizer(&sys_config) };
        if ptr.is_null() {
            None
        } else {
            Some(Self { ptr })
        }
    }

    /// Create an empty offline stream.
    pub fn create_stream(&self) -> OfflineStream {
        let ptr = unsafe { sys::SherpaOnnxCreateOfflineStream(self.ptr) };
        OfflineStream { ptr }
    }

    /// Create a stream with per-stream hotwords.
    pub fn create_stream_with_hotwords(&self, hotwords: &str) -> OfflineStream {
        let c = CString::new(hotwords).unwrap();
        let ptr = unsafe { sys::SherpaOnnxCreateOfflineStreamWithHotwords(self.ptr, c.as_ptr()) };
        OfflineStream { ptr }
    }

    /// Decode one stream.
    pub fn decode(&self, stream: &OfflineStream) {
        unsafe { sys::SherpaOnnxDecodeOfflineStream(self.ptr, stream.ptr) }
    }

    /// Decode multiple streams in one batch call.
    pub fn decode_multiple_streams(&self, streams: &[&OfflineStream]) {
        let ptrs: Vec<*const sys::OfflineStream> = streams
            .iter()
            .map(|s| s.ptr)
            .collect();
        unsafe {
            sys::SherpaOnnxDecodeMultipleOfflineStreams(self.ptr, ptrs.as_ptr(), ptrs.len() as i32)
        }
    }
}

impl Drop for OfflineRecognizer {
    fn drop(&mut self) {
        unsafe {
            sys::SherpaOnnxDestroyOfflineRecognizer(self.ptr);
        }
    }
}

/// Input stream used by [`OfflineRecognizer`].
pub struct OfflineStream {
    pub(crate) ptr: *const sys::OfflineStream,
}

impl OfflineStream {
    /// Append samples to the stream.
    pub fn accept_waveform(&self, sample_rate: i32, samples: &[f32]) {
        unsafe {
            sys::SherpaOnnxAcceptWaveformOffline(
                self.ptr,
                sample_rate,
                samples.as_ptr(),
                samples.len() as i32,
            )
        }
    }

    /// Fetch the current recognition result.
    pub fn get_result(&self) -> Option<OfflineRecognizerResult> {
        unsafe {
            let cstr = sys::SherpaOnnxGetOfflineStreamResultAsJson(self.ptr);
            if cstr.is_null() {
                return None;
            }
            let s = CStr::from_ptr(cstr)
                .to_string_lossy()
                .into_owned();
            sys::SherpaOnnxDestroyOfflineStreamResultJson(cstr);
            serde_json::from_str(&s).ok()
        }
    }

    pub fn set_option(&self, key: &str, value: &str) {
        let key = CString::new(key).unwrap();
        let value = CString::new(value).unwrap();
        unsafe { sys::SherpaOnnxOfflineStreamSetOption(self.ptr, key.as_ptr(), value.as_ptr()) }
    }

    pub fn get_option(&self, key: &str) -> String {
        let key = CString::new(key).unwrap();
        unsafe {
            let p = sys::SherpaOnnxOfflineStreamGetOption(self.ptr, key.as_ptr());
            if p.is_null() {
                String::new()
            } else {
                CStr::from_ptr(p)
                    .to_string_lossy()
                    .into_owned()
            }
        }
    }

    pub fn has_option(&self, key: &str) -> bool {
        let key = CString::new(key).unwrap();
        unsafe { sys::SherpaOnnxOfflineStreamHasOption(self.ptr, key.as_ptr()) != 0 }
    }
}

impl Drop for OfflineStream {
    fn drop(&mut self) {
        unsafe { sys::SherpaOnnxDestroyOfflineStream(self.ptr) }
    }
}
