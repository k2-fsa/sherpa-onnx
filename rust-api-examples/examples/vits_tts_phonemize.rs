// Copyright (c) 2026 Xiaomi Corporation
//
// This file demonstrates how to use a Piper VITS TTS model with sherpa-onnx's
// Rust API for offline text-to-speech.
//
// The text is phonemized outside of sherpa-onnx with the piper-phonemize crate
// and the resulting phoneme codepoints are passed via
// GenerationConfig::phoneme_codepoints. No lexicon file is needed.
//
// The Rust counterpart of
// https://github.com/k2-fsa/sherpa-onnx/blob/master/python-api-examples/test-offline-tts-piper-phonemize.py

use sherpa_onnx::{
    GenerationConfig, OfflineTts, OfflineTtsConfig, OfflineTtsVitsModelConfig,
};
use std::time::Instant;

fn main() {
    println!("piper-phonemize version: {}", piper_phonemize::get_version());

    // Note: the espeak-ng-data used by piper-phonemize is embedded in the
    // crate and initialized automatically on first use. It has nothing to do
    // with sherpa-onnx; sherpa-onnx itself does not use espeak-ng anymore.

    let config = OfflineTtsConfig {
        model: sherpa_onnx::OfflineTtsModelConfig {
            vits: OfflineTtsVitsModelConfig {
                model: Some("./vits-piper-en_US-amy-low/en_US-amy-low.onnx".into()),
                tokens: Some("./vits-piper-en_US-amy-low/tokens.txt".into()),
                noise_scale: 0.667,
                noise_scale_w: 0.8,
                length_scale: 1.0,
                ..Default::default()
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

    let text = "Liliana, the most beautiful and lovely assistant of our team!";

    // The espeak-ng voice for the phonemizer is taken from the model metadata.
    // Fall back to en-us if the model has no language information.
    let lang = {
        let l = tts.lang();
        if l.is_empty() {
            eprintln!(
                "Warning: the model has no language information. Falling back to en-us."
            );
            "en-us".to_string()
        } else {
            l
        }
    };
    println!("Phonemizer language: {}", lang);

    let result =
        piper_phonemize::phonemize(text, &lang).expect("Failed to phonemize the text");

    let mut phoneme_codepoints = Vec::new();
    for i in 0..result.num_sentences() {
        let phonemes = result.get_phonemes(i).unwrap_or_default();
        phoneme_codepoints.push(phonemes.into_iter().map(|c| c as i32).collect::<Vec<i32>>());
    }
    println!("Number of phonemized sentences: {}", phoneme_codepoints.len());

    let gen_config = GenerationConfig {
        sid: 0,
        speed: 1.0,
        phoneme_codepoints: Some(phoneme_codepoints),
        ..Default::default()
    };

    let start = Instant::now();

    // The text has already been phonemized; pass an empty string here.
    let audio = tts
        .generate_with_config(
            "",
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

    let filename = "./generated-vits-en-phonemize-rust.wav";
    if audio.save(filename) {
        println!("Saved to: {}", filename);
    } else {
        eprintln!("Failed to save {}", filename);
    }
}
