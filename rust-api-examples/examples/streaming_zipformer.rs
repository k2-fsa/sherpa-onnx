// Copyright (c)  2026  Xiaomi Corporation
//
// This file demonstrates how to use streaming Zipformer with sherpa-onnx's
// Rust API for speech recognition.
//
// See ../README.md for how to run it
//
// Note that even if we use a wave file as an example, this model supports
// real-time streaming speech recognition.
// See ./streaming_zipformer_microphone.rs for how to do real-time
// streaming speech recognition from a microphone.

use clap::Parser;
use sherpa_onnx::{OnlineRecognizer, OnlineRecognizerConfig, Wave};

/// Simple streaming Zipformer example
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

    /// Path to joiner ONNX model
    #[arg(long)]
    joiner: String,

    /// Path to tokens file
    #[arg(long)]
    tokens: String,

    /// Provider (default: cpu)
    #[arg(long, default_value = "cpu")]
    provider: String,

    /// Enable debug logs
    #[arg(long, default_value_t = false)]
    debug: bool,
}

fn main() {
    let args = Args::parse();

    let wave = Wave::read(&args.wav).expect("Failed to read WAV file");

    let mut recognizer_config = OnlineRecognizerConfig::default();
    recognizer_config.model_config.transducer.encoder = Some(args.encoder.clone());
    recognizer_config.model_config.transducer.decoder = Some(args.decoder.clone());
    recognizer_config.model_config.transducer.joiner = Some(args.joiner.clone());
    recognizer_config.model_config.tokens = Some(args.tokens.clone());
    recognizer_config.model_config.provider = Some(args.provider.clone());
    recognizer_config.enable_endpoint = true;
    recognizer_config.model_config.debug = args.debug;
    recognizer_config.decoding_method = Some("greedy_search".to_string());

    let recognizer =
        OnlineRecognizer::create(&recognizer_config).expect("Failed to create OnlineRecognizer");

    let stream = recognizer.create_stream();
    let mut segment_id = 0;

    // use any positive value as you like
    const CHUNK_SIZE: usize = 3200;

    println!(
        "Sample rate: {}, num samples: {}, duration: {:.2}s",
        wave.sample_rate(),
        wave.num_samples(),
        wave.num_samples() as f32 / wave.sample_rate() as f32
    );

    // Process in chunks
    for chunk in wave.samples().chunks(CHUNK_SIZE) {
        stream.accept_waveform(wave.sample_rate(), chunk);

        while recognizer.is_ready(&stream) {
            recognizer.decode(&stream);

            if let Some(result) = recognizer.get_result(&stream) {
                if !result.text.is_empty() {
                    println!("Segment {}: {}", segment_id, result.text);
                }
            }

            if recognizer.is_endpoint(&stream) {
                recognizer.reset(&stream);
                segment_id += 1;
            }
        }
    }

    // Tail padding (~0.3s)
    let tail_padding_len = (wave.sample_rate() as f32 * 0.3).round() as usize;
    let tail_padding = vec![0.0f32; tail_padding_len];
    stream.accept_waveform(wave.sample_rate(), &tail_padding);

    stream.input_finished();

    while recognizer.is_ready(&stream) {
        recognizer.decode(&stream);
        if let Some(result) = recognizer.get_result(&stream) {
            if !result.text.is_empty() {
                println!("Segment {}: {}", segment_id, result.text);
            }
        }
    }

    println!("Transcription finished.");
}
