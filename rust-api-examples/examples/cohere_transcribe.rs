// Copyright (c) 2026 Xiaomi Corporation
//
// This file demonstrates how to use Cohere Transcribe with sherpa-onnx's Rust
// API for offline speech recognition.

use clap::Parser;
use sherpa_onnx::{
    OfflineCohereTranscribeModelConfig, OfflineRecognizer, OfflineRecognizerConfig, Wave,
};

/// Cohere Transcribe offline example
#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Path to WAV file
    #[arg(long)]
    wav: String,

    /// Path to encoder ONNX model
    #[arg(long)]
    encoder: String,

    /// Path to decoder ONNX model
    #[arg(long)]
    decoder: String,

    /// Path to tokens.txt
    #[arg(long)]
    tokens: String,

    /// Language to decode
    #[arg(long, default_value = "en")]
    language: String,

    /// Provider (default: cpu)
    #[arg(long, default_value = "cpu")]
    provider: String,

    /// Enable debug logs
    #[arg(long, default_value_t = false)]
    debug: bool,

    /// Number of threads
    #[arg(long, default_value_t = 2)]
    num_threads: i32,
}

fn main() {
    let args = Args::parse();

    let wave = Wave::read(&args.wav).expect("Failed to read WAV file");

    let mut recognizer_config = OfflineRecognizerConfig::default();

    recognizer_config.model_config.cohere_transcribe = OfflineCohereTranscribeModelConfig {
        encoder: Some(args.encoder.clone()),
        decoder: Some(args.decoder.clone()),
        use_punct: true,
        use_itn: true,
        ..Default::default()
    };

    recognizer_config.model_config.tokens = Some(args.tokens.clone());
    recognizer_config.model_config.provider = Some(args.provider.clone());
    recognizer_config.model_config.debug = args.debug;
    recognizer_config.model_config.num_threads = args.num_threads;

    let recognizer =
        OfflineRecognizer::create(&recognizer_config).expect("Failed to create OfflineRecognizer");
    let stream = recognizer.create_stream();
    stream.set_option("language", &args.language);
    stream.accept_waveform(wave.sample_rate(), wave.samples());
    recognizer.decode(&stream);

    if let Some(result) = stream.get_result() {
        println!("Decoded text: {}", result.text);
    } else {
        eprintln!("Failed to get recognition result");
    }
}
