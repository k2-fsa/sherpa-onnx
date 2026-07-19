// Copyright (c)  2026  Stan Shih
//
// This file demonstrates how to use FunASR Nano with sherpa-onnx's Rust API
// for offline speech recognition.
//
// See also c-api-examples/funasr-nano-c-api.c and issue #3210.
// See ../README.md for how to run it.

use clap::Parser;
use sherpa_onnx::{
    OfflineFunASRNanoModelConfig, OfflineRecognizer, OfflineRecognizerConfig, Wave,
};
use std::time::Instant;

/// FunASR Nano offline ASR example
#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Path to input WAV file
    #[arg(long)]
    wav: String,

    /// Path to encoder_adaptor ONNX model
    #[arg(long)]
    encoder_adaptor: String,

    /// Path to embedding ONNX model
    #[arg(long)]
    embedding: String,

    /// Path to llm ONNX model
    #[arg(long)]
    llm: String,

    /// Path to tokenizer directory (e.g. Qwen3-0.6B)
    #[arg(long)]
    tokenizer: String,

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
    let audio_duration = wave.samples().len() as f64 / wave.sample_rate() as f64;

    let mut recognizer_config = OfflineRecognizerConfig::default();

    recognizer_config.model_config.funasr_nano = OfflineFunASRNanoModelConfig {
        encoder_adaptor: Some(args.encoder_adaptor.clone()),
        embedding: Some(args.embedding.clone()),
        llm: Some(args.llm.clone()),
        tokenizer: Some(args.tokenizer.clone()),
        ..Default::default()
    };

    recognizer_config.model_config.provider = Some(args.provider.clone());
    recognizer_config.model_config.debug = args.debug;
    recognizer_config.model_config.num_threads = args.num_threads;
    recognizer_config.decoding_method = Some("greedy_search".to_string());

    println!("Creating recognizer ...");
    let start_creation = Instant::now();
    let recognizer =
        OfflineRecognizer::create(&recognizer_config).expect("Failed to create OfflineRecognizer");
    let creation_elapsed = start_creation.elapsed().as_secs_f64();
    println!("Recognizer created in {:.3} seconds.", creation_elapsed);

    let stream = recognizer.create_stream();

    let start_recognition = Instant::now();
    stream.accept_waveform(wave.sample_rate(), wave.samples());
    recognizer.decode(&stream);
    let recognition_elapsed = start_recognition.elapsed().as_secs_f64();

    if let Some(result) = stream.get_result() {
        println!("Decoded text: {}", result.text);

        let total_elapsed = creation_elapsed + recognition_elapsed;
        let rtf = recognition_elapsed / audio_duration;
        println!("\n=== Performance Summary ===");
        println!("Audio duration          : {:.3} seconds", audio_duration);
        println!("Recognizer creation time: {:.3} seconds", creation_elapsed);
        println!(
            "Recognition time        : {:.3} seconds",
            recognition_elapsed
        );
        println!("Total elapsed time      : {:.3} seconds", total_elapsed);
        println!(
            "Real-Time Factor (RTF)  : {:.3} (recognition_elapsed / audio_duration = {:.3} / {:.3})",
            rtf, recognition_elapsed, audio_duration
        );
        println!(
            "Number of threads       : {}",
            recognizer_config.model_config.num_threads
        );
    } else {
        eprintln!("No recognition result");
    }
}
