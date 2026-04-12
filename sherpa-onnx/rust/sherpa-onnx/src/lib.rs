//! Safe Rust bindings for the public sherpa-onnx inference APIs.
//!
//! This crate wraps the sherpa-onnx C API with RAII-owned Rust types and
//! idiomatic configuration structs. The main feature families are:
//!
//! - offline ASR through [`OfflineRecognizer`]
//! - streaming ASR through [`OnlineRecognizer`]
//! - offline text-to-speech through [`OfflineTts`]
//! - voice activity detection through [`VoiceActivityDetector`]
//! - speaker embeddings and diarization
//! - online punctuation
//! - offline and streaming speech denoising
//! - audio tagging
//! - WAV I/O helpers through [`Wave`] and [`write()`]
//!
//! # Setup
//!
//! This crate now links statically by default. If `SHERPA_ONNX_LIB_DIR` is not
//! set, the build script downloads a matching prebuilt `-lib` archive from
//! [GitHub releases](https://github.com/k2-fsa/sherpa-onnx/releases) and uses
//! it automatically during the build.
//!
//! In other words, the default setup for most users is simply:
//!
//! ```toml
//! sherpa-onnx = "1.12.38"
//! ```
//!
//! If you want shared libraries instead, disable the default feature and enable
//! `shared`:
//!
//! ```toml
//! sherpa-onnx = { version = "1.12.38", default-features = false, features = ["shared"] }
//! ```
//!
//! For advanced use cases, set `SHERPA_ONNX_LIB_DIR` to a directory that already
//! contains sherpa-onnx libraries:
//!
//! ```bash
//! export SHERPA_ONNX_LIB_DIR=/path/to/sherpa-onnx/lib
//! ```
//!
//! That override works for both static and shared builds.
//!
//! Shared mode is also intended to work out of the box for normal users:
//!
//! - Linux and macOS: the build script adds both absolute and relative rpath
//!   entries automatically, and copies the required shared runtime libraries
//!   next to Cargo-generated binaries and examples.
//! - Windows: the build script copies the required DLLs next to the generated
//!   binaries automatically when using shared libraries.
//!
//! So most users do not need to manually set `LD_LIBRARY_PATH` or
//! `DYLD_LIBRARY_PATH`.
//!
//! Example `v1.12.38` archives used by the build script:
//!
//! Default static archives:
//!
//! - Linux x86_64:
//!   [sherpa-onnx-v1.12.38-linux-x64-static-lib.tar.bz2](https://github.com/k2-fsa/sherpa-onnx/releases/download/v1.12.38/sherpa-onnx-v1.12.38-linux-x64-static-lib.tar.bz2)
//! - Linux aarch64:
//!   [sherpa-onnx-v1.12.38-linux-aarch64-static-lib.tar.bz2](https://github.com/k2-fsa/sherpa-onnx/releases/download/v1.12.38/sherpa-onnx-v1.12.38-linux-aarch64-static-lib.tar.bz2)
//! - macOS x86_64:
//!   [sherpa-onnx-v1.12.38-osx-x64-static-lib.tar.bz2](https://github.com/k2-fsa/sherpa-onnx/releases/download/v1.12.38/sherpa-onnx-v1.12.38-osx-x64-static-lib.tar.bz2)
//! - macOS arm64:
//!   [sherpa-onnx-v1.12.38-osx-arm64-static-lib.tar.bz2](https://github.com/k2-fsa/sherpa-onnx/releases/download/v1.12.38/sherpa-onnx-v1.12.38-osx-arm64-static-lib.tar.bz2)
//! - Windows x64:
//!   [sherpa-onnx-v1.12.38-win-x64-static-MT-Release-lib.tar.bz2](https://github.com/k2-fsa/sherpa-onnx/releases/download/v1.12.38/sherpa-onnx-v1.12.38-win-x64-static-MT-Release-lib.tar.bz2)
//!
//! Optional shared archives:
//!
//! - Linux x86_64:
//!   [sherpa-onnx-v1.12.38-linux-x64-shared-lib.tar.bz2](https://github.com/k2-fsa/sherpa-onnx/releases/download/v1.12.38/sherpa-onnx-v1.12.38-linux-x64-shared-lib.tar.bz2)
//! - Linux aarch64:
//!   [sherpa-onnx-v1.12.38-linux-aarch64-shared-cpu-lib.tar.bz2](https://github.com/k2-fsa/sherpa-onnx/releases/download/v1.12.38/sherpa-onnx-v1.12.38-linux-aarch64-shared-cpu-lib.tar.bz2)
//! - macOS x86_64:
//!   [sherpa-onnx-v1.12.38-osx-x64-shared-lib.tar.bz2](https://github.com/k2-fsa/sherpa-onnx/releases/download/v1.12.38/sherpa-onnx-v1.12.38-osx-x64-shared-lib.tar.bz2)
//! - macOS arm64:
//!   [sherpa-onnx-v1.12.38-osx-arm64-shared-lib.tar.bz2](https://github.com/k2-fsa/sherpa-onnx/releases/download/v1.12.38/sherpa-onnx-v1.12.38-osx-arm64-shared-lib.tar.bz2)
//! - Windows x64:
//!   [sherpa-onnx-v1.12.38-win-x64-shared-MT-Release-lib.tar.bz2](https://github.com/k2-fsa/sherpa-onnx/releases/download/v1.12.38/sherpa-onnx-v1.12.38-win-x64-shared-MT-Release-lib.tar.bz2)
//!
//! # How the Rust API is organized
//!
//! Most APIs follow the same pattern:
//!
//! 1. Start with a `*Config` value and fill the fields for exactly one model
//!    family.
//! 2. Call `create()` to construct the runtime object.
//! 3. Create a stream if the API is stream-based.
//! 4. Feed audio or text, then fetch results with the provided accessor methods.
//!
//! All runtime wrappers automatically free their underlying C resources on drop.
//!
//! # Examples
//!
//! The repository contains end-to-end Rust examples under
//! [`rust-api-examples/examples/`](https://github.com/k2-fsa/sherpa-onnx/tree/master/rust-api-examples/examples).
//! Good entry points are:
//!
//! - [`sense_voice.rs`](https://github.com/k2-fsa/sherpa-onnx/blob/master/rust-api-examples/examples/sense_voice.rs)
//! - [`nemo_parakeet.rs`](https://github.com/k2-fsa/sherpa-onnx/blob/master/rust-api-examples/examples/nemo_parakeet.rs)
//! - [`streaming_zipformer.rs`](https://github.com/k2-fsa/sherpa-onnx/blob/master/rust-api-examples/examples/streaming_zipformer.rs)
//! - [`pocket_tts.rs`](https://github.com/k2-fsa/sherpa-onnx/blob/master/rust-api-examples/examples/pocket_tts.rs)
//! - [`silero_vad_remove_silence.rs`](https://github.com/k2-fsa/sherpa-onnx/blob/master/rust-api-examples/examples/silero_vad_remove_silence.rs)
//! - [`online_punctuation.rs`](https://github.com/k2-fsa/sherpa-onnx/blob/master/rust-api-examples/examples/online_punctuation.rs)
//! - [`offline_punctuation.rs`](https://github.com/k2-fsa/sherpa-onnx/blob/master/rust-api-examples/examples/offline_punctuation.rs)
//! - [`keyword_spotter.rs`](https://github.com/k2-fsa/sherpa-onnx/blob/master/rust-api-examples/examples/keyword_spotter.rs)
//! - [`spoken_language_identification.rs`](https://github.com/k2-fsa/sherpa-onnx/blob/master/rust-api-examples/examples/spoken_language_identification.rs)
//! - [`offline_speaker_diarization.rs`](https://github.com/k2-fsa/sherpa-onnx/blob/master/rust-api-examples/examples/offline_speaker_diarization.rs)
//! - [`speaker_embedding_manager.rs`](https://github.com/k2-fsa/sherpa-onnx/blob/master/rust-api-examples/examples/speaker_embedding_manager.rs)
//!
//! # Offline recognition example
//!
//! ```no_run
//! use sherpa_onnx::{
//!     OfflineRecognizer, OfflineRecognizerConfig, OfflineSenseVoiceModelConfig, Wave,
//! };
//!
//! let wave = Wave::read("./test.wav").expect("read wave");
//!
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
//!
//! let result = stream.get_result().expect("result");
//! println!("{}", result.text);
//! ```
//!
//! # Streaming recognition example
//!
//! ```no_run
//! use sherpa_onnx::{OnlineRecognizer, OnlineRecognizerConfig, Wave};
//!
//! let wave = Wave::read("./test.wav").expect("read wave");
//!
//! let mut config = OnlineRecognizerConfig::default();
//! config.model_config.transducer.encoder = Some(
//!     "./sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20/encoder-epoch-99-avg-1.int8.onnx".into(),
//! );
//! config.model_config.transducer.decoder = Some(
//!     "./sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20/decoder-epoch-99-avg-1.onnx".into(),
//! );
//! config.model_config.transducer.joiner = Some(
//!     "./sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20/joiner-epoch-99-avg-1.int8.onnx".into(),
//! );
//! config.model_config.tokens = Some(
//!     "./sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20/tokens.txt".into(),
//! );
//! config.enable_endpoint = true;
//! config.decoding_method = Some("greedy_search".into());
//!
//! let recognizer = OnlineRecognizer::create(&config).expect("create recognizer");
//! let stream = recognizer.create_stream();
//! stream.accept_waveform(wave.sample_rate(), wave.samples());
//! stream.input_finished();
//! while recognizer.is_ready(&stream) {
//!     recognizer.decode(&stream);
//! }
//! ```
//!
//! # TTS example
//!
//! ```no_run
//! use sherpa_onnx::{OfflineTts, OfflineTtsConfig, OfflineTtsModelConfig, OfflineTtsPocketModelConfig};
//!
//! let config = OfflineTtsConfig {
//!     model: OfflineTtsModelConfig {
//!         pocket: OfflineTtsPocketModelConfig {
//!             lm_flow: Some("./sherpa-onnx-pocket-tts-int8-2026-01-26/lm_flow.int8.onnx".into()),
//!             lm_main: Some("./sherpa-onnx-pocket-tts-int8-2026-01-26/lm_main.int8.onnx".into()),
//!             encoder: Some("./sherpa-onnx-pocket-tts-int8-2026-01-26/encoder.onnx".into()),
//!             decoder: Some("./sherpa-onnx-pocket-tts-int8-2026-01-26/decoder.int8.onnx".into()),
//!             text_conditioner: Some(
//!                 "./sherpa-onnx-pocket-tts-int8-2026-01-26/text_conditioner.onnx".into(),
//!             ),
//!             vocab_json: Some("./sherpa-onnx-pocket-tts-int8-2026-01-26/vocab.json".into()),
//!             token_scores_json: Some(
//!                 "./sherpa-onnx-pocket-tts-int8-2026-01-26/token_scores.json".into(),
//!             ),
//!             ..Default::default()
//!         },
//!         ..Default::default()
//!     },
//!     ..Default::default()
//! };
//!
//! let tts = OfflineTts::create(&config).expect("create tts");
//! println!("{}", tts.sample_rate());
//! ```
mod audio_tagging;
mod display;
mod kws;
mod offline_asr;
mod offline_punctuation;
mod offline_speaker_diarization;
mod offline_speech_denoiser;
mod online_asr;
mod online_punctuation;
mod online_speech_denoiser;
mod resampler;
mod speaker_embedding;
mod spoken_language_identification;
mod tts;
mod utils;
mod vad;
mod wave;

pub use audio_tagging::*;
pub use display::*;
pub use kws::*;
pub use offline_asr::*;
pub use offline_punctuation::*;
pub use offline_speaker_diarization::*;
pub use offline_speech_denoiser::*;
pub use online_asr::*;
pub use online_punctuation::*;
pub use online_speech_denoiser::*;
pub use resampler::*;
pub use speaker_embedding::*;
pub use spoken_language_identification::*;
pub use tts::*;
pub use utils::*;
pub use vad::*;
pub use wave::*;
