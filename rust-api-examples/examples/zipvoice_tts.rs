// Copyright (c) 2026 Xiaomi Corporation
//
// This file demonstrates how to use ZipVoice TTS with sherpa-onnx's Rust API
// for offline zero-shot text-to-speech.

use sherpa_onnx::{
    GenerationConfig, OfflineTts, OfflineTtsConfig, OfflineTtsZipvoiceModelConfig, Wave,
};
use std::collections::HashMap;
use std::time::Instant;

fn main() {
    let config = OfflineTtsConfig {
        model: sherpa_onnx::OfflineTtsModelConfig {
            zipvoice: OfflineTtsZipvoiceModelConfig {
                tokens: Some("./sherpa-onnx-zipvoice-distill-int8-zh-en-emilia/tokens.txt".into()),
                encoder: Some(
                    "./sherpa-onnx-zipvoice-distill-int8-zh-en-emilia/encoder.int8.onnx".into(),
                ),
                decoder: Some(
                    "./sherpa-onnx-zipvoice-distill-int8-zh-en-emilia/decoder.int8.onnx".into(),
                ),
                vocoder: Some("./vocos_24khz.onnx".into()),
                data_dir: Some(
                    "./sherpa-onnx-zipvoice-distill-int8-zh-en-emilia/espeak-ng-data".into(),
                ),
                lexicon: Some(
                    "./sherpa-onnx-zipvoice-distill-int8-zh-en-emilia/lexicon.txt".into(),
                ),
                feat_scale: 0.1,
                t_shift: 0.5,
                target_rms: 0.1,
                guidance_scale: 1.0,
            },
            num_threads: 2,
            debug: false,
            ..Default::default()
        },
        ..Default::default()
    };

    let tts = OfflineTts::create(&config).expect("Failed to create OfflineTts");

    println!("Sample rate: {}", tts.sample_rate());
    println!("Num speakers: {}", tts.num_speakers());

    let text = "小米的价值观是真诚, 热爱. 真诚，就是不欺人也不自欺. 热爱, 就是全心投入并享受其中.";
    let reference_text = "那还是三十六年前, 一九八七年. 我呢考上了武汉大学的计算机系.";
    let reference_audio_file =
        "./sherpa-onnx-zipvoice-distill-int8-zh-en-emilia/test_wavs/leijun-1.wav";

    let wave = Wave::read(reference_audio_file).expect("Failed to read reference audio");

    let mut extra = HashMap::new();
    extra.insert("min_char_in_sentence".to_string(), serde_json::json!(10));

    let gen_config = GenerationConfig {
        speed: 1.0,
        reference_audio: Some(wave.samples().to_vec()),
        reference_sample_rate: wave.sample_rate(),
        reference_text: Some(reference_text.to_string()),
        num_steps: 4,
        extra: Some(extra),
        ..Default::default()
    };

    let start = Instant::now();

    let audio = tts
        .generate_with_config(
            text,
            &gen_config,
            Some(|_samples: &[f32], progress: f32| -> bool {
                println!("Progress: {:.1}%", progress * 100.0);
                true
            }),
        )
        .expect("Generation failed");

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

    let filename = "./generated-zipvoice-zh-en-rust.wav";
    if audio.save(filename) {
        println!("Saved to: {}", filename);
    } else {
        eprintln!("Failed to save {}", filename);
    }
}
