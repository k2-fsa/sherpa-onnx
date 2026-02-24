// Copyright (c) 2026 Xiaomi Corporation
//
// This file demonstrates how to use silero VAD with sherpa-onnx's
// Rust API to remove non-speech segments and save speech-only audio.
//
// See ../README.md for how to run it

use clap::Parser;
use sherpa_onnx::{self, SileroVadModelConfig, VadModelConfig, VoiceActivityDetector, Wave};

/// Simple VAD example: remove non-speech segments from a WAV file
#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Path to input WAV file
    #[arg(long)]
    input: String,

    /// Path to output WAV file
    #[arg(long)]
    output: String,

    /// Path to Silero VAD ONNX model
    #[arg(long)]
    silero_vad_model: String,
}

fn main() -> anyhow::Result<()> {
    let args = Args::parse();

    // Read WAV file
    let wave = Wave::read(&args.input)
        .ok_or_else(|| anyhow::anyhow!("Failed to read WAV file: {}", &args.input))?;
    let sample_rate = wave.sample_rate();
    let input_num_samples = wave.num_samples();
    let input_duration = input_num_samples as f32 / sample_rate as f32;

    println!(
        "Input WAV: sample rate: {}, num samples: {}, duration: {:.2}s",
        sample_rate, input_num_samples, input_duration
    );

    // Configure VAD
    let mut silero_config = SileroVadModelConfig::default();
    silero_config.model = Some(args.silero_vad_model);

    // You can tune the values below
    silero_config.threshold = 0.5;
    silero_config.min_silence_duration = 0.25;
    silero_config.min_speech_duration = 0.25;
    silero_config.max_speech_duration = 5.0;

    let vad_config = VadModelConfig {
        silero_vad: silero_config,
        ten_vad: Default::default(),
        sample_rate,
        num_threads: 1,
        provider: Some("cpu".to_string()),
        debug: false,
    };

    let vad = VoiceActivityDetector::create(&vad_config, 30.0)
        .expect("Failed to create VoiceActivityDetector");

    let mut speech_samples = Vec::new();
    const WINDOW_SIZE: usize = 512;

    for chunk in wave.samples().chunks(WINDOW_SIZE) {
        vad.accept_waveform(chunk);

        while let Some(seg) = vad.front() {
            speech_samples.extend_from_slice(seg.samples());
            vad.pop();
        }
    }

    vad.flush();
    while let Some(seg) = vad.front() {
        speech_samples.extend_from_slice(seg.samples());
        vad.pop();
    }

    // Write speech-only samples to output WAV
    let ok = sherpa_onnx::write(&args.output, &speech_samples, sample_rate);
    if ok {
        println!("Saved speech-only audio to {}", args.output);
    } else {
        println!("Failed to save speech-only audio to {}", args.output);
    }

    // Summary
    let output_num_samples = speech_samples.len();
    let output_duration = output_num_samples as f32 / sample_rate as f32;
    println!("\n=== Summary ===");
    println!(
        "Input:  sample rate = {}, samples = {}, duration = {:.2}s",
        sample_rate, input_num_samples, input_duration
    );
    println!(
        "Output: sample rate = {}, samples = {}, duration = {:.2}s",
        sample_rate, output_num_samples, output_duration
    );
    println!(
        "Removed non-speech: {:.2}% of input removed",
        100.0 * (1.0 - output_duration / input_duration)
    );

    Ok(())
}
