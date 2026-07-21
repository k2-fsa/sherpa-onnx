// Copyright (c) 2026 Xiaomi Corporation
//
// This file demonstrates how to use ten-vad with sherpa-onnx's
// Rust API to remove non-speech segments and save speech-only audio.
//
// See ../README.md for how to run it

use clap::Parser;
use sherpa_onnx::{self, TenVadModelConfig, VadModelConfig, VoiceActivityDetector, Wave};

/// Simple VAD example: remove non-speech segments from a WAV file using ten-vad
#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Path to input WAV file
    #[arg(long)]
    input: String,

    /// Path to output WAV file
    #[arg(long)]
    output: String,

    /// Path to ten-vad ONNX model
    #[arg(long)]
    ten_vad_model: String,
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
    let mut ten_vad_config = TenVadModelConfig::default();
    ten_vad_config.model = Some(args.ten_vad_model);

    // ten-vad expects a window size of 256 samples at 16 kHz.
    // Please don't change it unless you know what you are doing.
    ten_vad_config.window_size = 256;

    // You can tune the values below
    ten_vad_config.threshold = 0.5;
    ten_vad_config.min_silence_duration = 0.25;
    ten_vad_config.min_speech_duration = 0.5;
    ten_vad_config.max_speech_duration = 5.0;

    let window_size = ten_vad_config.window_size as usize;

    let vad_config = VadModelConfig {
        silero_vad: Default::default(),
        ten_vad: ten_vad_config,
        sample_rate,
        num_threads: 1,
        provider: Some("cpu".to_string()),
        debug: false,
    };

    let vad = VoiceActivityDetector::create(&vad_config, 30.0)
        .ok_or_else(|| anyhow::anyhow!("Failed to create VoiceActivityDetector"))?;

    let mut speech_samples = Vec::new();

    for chunk in wave.samples().chunks(window_size) {
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
    if !ok {
        anyhow::bail!("Failed to save speech-only audio to {}", args.output);
    }
    println!("Saved speech-only audio to {}", args.output);

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
    if input_duration > 0.0 {
        println!(
            "Removed non-speech: {:.2}% of input removed",
            100.0 * (1.0 - output_duration / input_duration)
        );
    }

    Ok(())
}
