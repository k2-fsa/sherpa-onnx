// Copyright (c) 2026 Xiaomi Corporation
//
// This file demonstrates how to use FireRedAsr CTC with sherpa-onnx's Rust API
// for offline speech recognition.
//
// See ../README.md for how to run it.

use clap::Parser;
use sherpa_onnx::{
    OfflineFireRedAsrCtcModelConfig, OfflineRecognizer, OfflineRecognizerConfig, Wave,
};
use std::time::Instant;

/// FireRedAsr CTC offline example
#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Path to WAV file
    #[arg(long)]
    wav: String,

    /// Path to FireRedAsr CTC ONNX model
    #[arg(long)]
    model: String,

    /// Path to tokens file
    #[arg(long)]
    tokens: String,

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

    recognizer_config.model_config.fire_red_asr_ctc = OfflineFireRedAsrCtcModelConfig {
        model: Some(args.model.clone()),
    };

    recognizer_config.model_config.tokens = Some(args.tokens.clone());
    recognizer_config.model_config.provider = Some(args.provider.clone());
    recognizer_config.model_config.debug = args.debug;
    recognizer_config.model_config.num_threads = args.num_threads;

    // Measure recognizer creation time
    println!("Creating recognizer ...");
    let start_creation = Instant::now();
    let recognizer =
        OfflineRecognizer::create(&recognizer_config).expect("Failed to create OfflineRecognizer");
    let creation_elapsed = start_creation.elapsed().as_secs_f64();
    println!("Recognizer created in {:.3} seconds.", creation_elapsed);

    let stream = recognizer.create_stream();

    // Measure recognition time
    let start_recognition = Instant::now();
    stream.accept_waveform(wave.sample_rate(), wave.samples());
    recognizer.decode(&stream);
    let recognition_elapsed = start_recognition.elapsed().as_secs_f64();

    // Get recognition result
    if let Some(result) = stream.get_result() {
        println!("Decoded text: {}", result.text);

        let total_time = creation_elapsed + recognition_elapsed;
        let rtf = recognition_elapsed / audio_duration;

        println!("\n=== Performance Summary ===");
        println!("Audio duration          : {:.3} seconds", audio_duration);
        println!("Recognizer creation time: {:.3} seconds", creation_elapsed);
        println!(
            "Recognition time        : {:.3} seconds",
            recognition_elapsed
        );
        println!("Total elapsed time      : {:.3} seconds", total_time);

        // Detailed RTF computation log
        println!(
            "Real-Time Factor (RTF)  : {:.3} (recognition_elapsed / audio_duration = {:.3} / {:.3})",
            rtf, recognition_elapsed, audio_duration
        );

        println!(
            "Number of threads       : {}",
            recognizer_config.model_config.num_threads
        );
    } else {
        eprintln!("Failed to get recognition result");
    }
}
