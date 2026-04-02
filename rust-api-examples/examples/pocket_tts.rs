// Copyright (c) 2026 Xiaomi Corporation
//
// This file demonstrates how to use Pocket TTS with sherpa-onnx's Rust API
// for offline text-to-speech with zero-shot voice cloning.

use sherpa_onnx::{
    GenerationConfig, OfflineTts, OfflineTtsConfig, OfflineTtsPocketModelConfig, Wave,
};
use std::collections::HashMap;
use std::time::Instant;

fn main() {
    let config = OfflineTtsConfig {
        model: sherpa_onnx::OfflineTtsModelConfig {
            pocket: OfflineTtsPocketModelConfig {
                lm_flow: Some("./sherpa-onnx-pocket-tts-int8-2026-01-26/lm_flow.int8.onnx".into()),
                lm_main: Some("./sherpa-onnx-pocket-tts-int8-2026-01-26/lm_main.int8.onnx".into()),
                encoder: Some("./sherpa-onnx-pocket-tts-int8-2026-01-26/encoder.onnx".into()),
                decoder: Some("./sherpa-onnx-pocket-tts-int8-2026-01-26/decoder.int8.onnx".into()),
                text_conditioner: Some(
                    "./sherpa-onnx-pocket-tts-int8-2026-01-26/text_conditioner.onnx".into(),
                ),
                vocab_json: Some("./sherpa-onnx-pocket-tts-int8-2026-01-26/vocab.json".into()),
                token_scores_json: Some(
                    "./sherpa-onnx-pocket-tts-int8-2026-01-26/token_scores.json".into(),
                ),
                voice_embedding_cache_capacity: 50,
            },
            num_threads: 2,
            debug: false, // set to true to see verbose logs
            ..Default::default()
        },
        ..Default::default()
    };

    let tts = OfflineTts::create(&config).expect("Failed to create OfflineTts");

    println!("Sample rate: {}", tts.sample_rate());
    println!("Num speakers: {}", tts.num_speakers());

    let text = "Today as always, men fall into two groups: slaves and free men. Whoever \
        does not have two-thirds of his day for himself, is a slave, whatever \
        he may be: a statesman, a businessman, an official, or a scholar. \
        Friends fell out often because life was changing so fast. The easiest \
        thing in the world was to lose touch with someone.";

    // Read reference audio for zero-shot voice cloning
    let reference_audio_file = "./sherpa-onnx-pocket-tts-int8-2026-01-26/test_wavs/bria.wav";
    let wave = Wave::read(reference_audio_file).expect("Failed to read reference audio");

    let mut extra = HashMap::new();
    extra.insert(
        "max_reference_audio_len".to_string(),
        serde_json::json!(10.0),
    );
    extra.insert("seed".to_string(), serde_json::json!(42));

    let gen_config = GenerationConfig {
        speed: 1.0,
        reference_audio: Some(wave.samples().to_vec()),
        reference_sample_rate: wave.sample_rate(),
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

    let filename = "./generated-pocket-en-rust.wav";
    if audio.save(filename) {
        println!("Saved to: {}", filename);
    } else {
        eprintln!("Failed to save {}", filename);
    }
}
