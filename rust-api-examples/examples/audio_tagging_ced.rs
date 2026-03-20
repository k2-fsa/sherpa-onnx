// Copyright (c) 2026 Xiaomi Corporation
//
// This file demonstrates how to use audio tagging with a CED model
// through sherpa-onnx's Rust API.

use sherpa_onnx::{AudioTagging, AudioTaggingConfig, AudioTaggingModelConfig, Wave};
use std::time::Instant;

fn main() {
    let config = AudioTaggingConfig {
        model: AudioTaggingModelConfig {
            ced: Some("./sherpa-onnx-ced-mini-audio-tagging-2024-04-19/model.int8.onnx".into()),
            num_threads: 1,
            debug: true,
            provider: Some("cpu".into()),
            ..Default::default()
        },
        labels: Some(
            "./sherpa-onnx-ced-mini-audio-tagging-2024-04-19/class_labels_indices.csv".into(),
        ),
        top_k: 5,
    };

    let tagger = AudioTagging::create(&config).expect("Failed to create AudioTagging");

    let wav = Wave::read("./sherpa-onnx-ced-mini-audio-tagging-2024-04-19/test_wavs/6.wav")
        .expect("Failed to read test wave");

    let start = Instant::now();
    let stream = tagger.create_stream();
    stream.accept_waveform(wav.sample_rate(), wav.samples());
    let result = tagger.compute(&stream, 5);
    let elapsed_seconds = start.elapsed().as_secs_f32();
    let audio_duration = wav.samples().len() as f32 / wav.sample_rate() as f32;
    let rtf = elapsed_seconds / audio_duration;

    println!("Elapsed seconds: {:.3}", elapsed_seconds);
    println!("Audio duration in seconds: {:.3}", audio_duration);
    println!("RTF: {:.3}/{:.3} = {:.3}", elapsed_seconds, audio_duration, rtf);
    println!();
    for (i, event) in result.iter().enumerate() {
        println!("{}: {{name: {}, index: {}, prob: {:.3}}}", i, event.name, event.index, event.prob);
    }
}
