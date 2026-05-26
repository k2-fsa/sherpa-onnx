// Copyright (c) 2026 Xiaomi Corporation
//
// This file demonstrates how to use sherpa-onnx's Rust API for spoken language
// identification.
//
// See ../README.md for how to run it.

use clap::Parser;
use sherpa_onnx::{
    SpokenLanguageIdentification, SpokenLanguageIdentificationConfig,
    SpokenLanguageIdentificationWhisperConfig, Wave,
};
use std::time::Instant;

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    #[arg(long)]
    wav: String,

    #[arg(long)]
    whisper_encoder: String,

    #[arg(long)]
    whisper_decoder: String,

    #[arg(long, default_value_t = 0)]
    tail_paddings: i32,

    #[arg(long, default_value_t = 1)]
    num_threads: i32,

    #[arg(long, default_value = "cpu")]
    provider: String,

    #[arg(long, default_value_t = false)]
    debug: bool,
}

fn main() -> anyhow::Result<()> {
    let args = Args::parse();
    let wave = Wave::read(&args.wav).ok_or_else(|| anyhow::anyhow!("Failed to read WAV file"))?;
    let audio_duration = wave.num_samples() as f64 / wave.sample_rate() as f64;

    let config = SpokenLanguageIdentificationConfig {
        whisper: SpokenLanguageIdentificationWhisperConfig {
            encoder: Some(args.whisper_encoder),
            decoder: Some(args.whisper_decoder),
            tail_paddings: args.tail_paddings,
        },
        num_threads: args.num_threads,
        provider: Some(args.provider),
        debug: args.debug,
    };

    let slid = SpokenLanguageIdentification::create(&config)
        .ok_or_else(|| anyhow::anyhow!("Failed to create SpokenLanguageIdentification"))?;

    let stream = slid.create_stream();
    let start = Instant::now();
    stream.accept_waveform(wave.sample_rate(), wave.samples());
    let result = slid
        .compute(&stream)
        .ok_or_else(|| anyhow::anyhow!("Failed to compute spoken language identification result"))?;
    let elapsed = start.elapsed().as_secs_f64();

    println!("File: {}", args.wav);
    println!("Detected language: {}", result.lang);
    println!("Elapsed seconds: {:.3}", elapsed);
    println!("Audio duration in seconds: {:.3}", audio_duration);
    println!("RTF: {:.3}/{:.3} = {:.3}", elapsed, audio_duration, elapsed / audio_duration);

    Ok(())
}
