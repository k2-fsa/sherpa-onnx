// Copyright (c) 2026 Xiaomi Corporation
//
// This file demonstrates how to use Supertonic TTS with sherpa-onnx's Rust API
// for offline text-to-speech.

use sherpa_onnx::{
    GenerationConfig, OfflineTts, OfflineTtsConfig, OfflineTtsSupertonicModelConfig,
};
use std::collections::HashMap;
use std::time::Instant;

fn main() {
    let config = OfflineTtsConfig {
        model: sherpa_onnx::OfflineTtsModelConfig {
            supertonic: OfflineTtsSupertonicModelConfig {
                duration_predictor: Some(
                    "./sherpa-onnx-supertonic-tts-int8-2026-03-06/duration_predictor.int8.onnx"
                        .into(),
                ),
                text_encoder: Some(
                    "./sherpa-onnx-supertonic-tts-int8-2026-03-06/text_encoder.int8.onnx".into(),
                ),
                vector_estimator: Some(
                    "./sherpa-onnx-supertonic-tts-int8-2026-03-06/vector_estimator.int8.onnx"
                        .into(),
                ),
                vocoder: Some(
                    "./sherpa-onnx-supertonic-tts-int8-2026-03-06/vocoder.int8.onnx".into(),
                ),
                tts_json: Some("./sherpa-onnx-supertonic-tts-int8-2026-03-06/tts.json".into()),
                unicode_indexer: Some(
                    "./sherpa-onnx-supertonic-tts-int8-2026-03-06/unicode_indexer.bin".into(),
                ),
                voice_style: Some("./sherpa-onnx-supertonic-tts-int8-2026-03-06/voice.bin".into()),
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
        he may be: a statesman, a businessman, an official, or a scholar.";

    let mut extra = HashMap::new();
    extra.insert("lang".to_string(), serde_json::json!("en"));

    let gen_config = GenerationConfig {
        sid: 6,
        num_steps: 5,
        speed: 1.25,
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

    let filename = "./generated-supertonic-en-rust.wav";
    if audio.save(filename) {
        println!("Saved to: {}", filename);
    } else {
        eprintln!("Failed to save {}", filename);
    }
}
