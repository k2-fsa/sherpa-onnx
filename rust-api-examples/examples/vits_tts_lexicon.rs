// Copyright (c) 2026 Xiaomi Corporation
//
// This file demonstrates how to use a Piper VITS TTS model with sherpa-onnx's
// Rust API for offline text-to-speech.

use clap::Parser;
use sherpa_onnx::{
    GenerationConfig, OfflineTts, OfflineTtsConfig, OfflineTtsVitsModelConfig,
};
use std::time::Instant;

#[derive(Parser, Debug)]
#[command(author, version, about)]
struct Args {
    /// Path to the VITS/Piper model
    #[arg(long)]
    model: String,

    /// Path to tokens.txt
    #[arg(long)]
    tokens: String,

    /// Path to lexicon.txt
    #[arg(long)]
    lexicon: String,

    /// Input text to synthesize
    #[arg(long)]
    text: String,

    /// Output wave filename
    #[arg(long, default_value = "./generated-vits-lexicon-rust.wav")]
    output: String,

    /// Speaker ID for multi-speaker models
    #[arg(long, default_value_t = 0)]
    sid: i32,

    /// Speech speed; larger means faster
    #[arg(long, default_value_t = 1.0)]
    speed: f32,

    /// Number of threads
    #[arg(long, default_value_t = 2)]
    num_threads: i32,

    /// Show debug logs from sherpa-onnx
    #[arg(long, default_value_t = false)]
    debug: bool,
}

fn main() {
    let args = Args::parse();

    let config = OfflineTtsConfig {
        model: sherpa_onnx::OfflineTtsModelConfig {
            vits: OfflineTtsVitsModelConfig {
                model: Some(args.model.clone()),
                tokens: Some(args.tokens.clone()),
                noise_scale: 0.667,
                noise_scale_w: 0.8,
                length_scale: 1.0,
                lexicon: Some(args.lexicon.clone()),
                ..Default::default()
            },
            num_threads: args.num_threads,
            debug: args.debug,
            ..Default::default()
        },
        ..Default::default()
    };

    if let Ok(maps) = std::fs::read_to_string("/proc/self/maps") {
        for line in maps.lines() {
            if line.contains("sherpa") || line.contains("onnxruntime") {
                eprintln!("[dbg] loaded: {}", line);
            }
        }
    }
    eprintln!("[dbg] before create");
    let tts = OfflineTts::create(&config).expect("Failed to create OfflineTts");
    eprintln!("[dbg] after create");

    println!("Sample rate: {}", tts.sample_rate());
    println!("Num speakers: {}", tts.num_speakers());

    let gen_config = GenerationConfig {
        sid: args.sid,
        speed: args.speed,
        ..Default::default()
    };

    let start = Instant::now();

    eprintln!("[dbg] before generate");
    let audio = tts
        .generate_with_config(
            &args.text,
            &gen_config,
            Some(|_samples: &[f32], progress: f32| -> bool {
                println!("Progress: {:.1}%", progress * 100.0);
                true
            }),
        )
        .expect("Generation failed");
    eprintln!("[dbg] after generate");

    let elapsed_seconds = start.elapsed().as_secs_f32();
    let duration = audio.samples().len() as f32 / audio.sample_rate() as f32;
    let rtf = elapsed_seconds / duration;

    println!("Number of threads: {}", config.model.num_threads);
    println!("Elapsed seconds: {:.3} s", elapsed_seconds);
    println!("Audio duration: {:.3} s", duration);
    println!(
        "Real-time factor (RTF): {:.3}/{:.3} = {:.3}",
        elapsed_seconds, duration, rtf
    );
    eprintln!("[dbg] before save");
    if audio.save(&args.output) {
        println!("Saved to: {}", args.output);
    } else {
        eprintln!("Failed to save {}", args.output);
    }
    eprintln!("[dbg] after save; dropping");
    drop(audio);
    drop(tts);
    eprintln!("[dbg] done");
}
