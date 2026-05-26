// Copyright (c) 2026 Xiaomi Corporation
//
// This file demonstrates how to use Matcha TTS with sherpa-onnx's Rust API
// for offline Chinese text-to-speech.

use sherpa_onnx::{
    GenerationConfig, OfflineTts, OfflineTtsConfig, OfflineTtsMatchaModelConfig,
};
use std::time::Instant;

fn main() {
    let config = OfflineTtsConfig {
        model: sherpa_onnx::OfflineTtsModelConfig {
            matcha: OfflineTtsMatchaModelConfig {
                acoustic_model: Some("./matcha-icefall-zh-baker/model-steps-3.onnx".into()),
                vocoder: Some("./vocos-22khz-univ.onnx".into()),
                lexicon: Some("./matcha-icefall-zh-baker/lexicon.txt".into()),
                tokens: Some("./matcha-icefall-zh-baker/tokens.txt".into()),
                dict_dir: Some("./matcha-icefall-zh-baker/dict".into()),
                noise_scale: 0.667,
                length_scale: 1.0,
                ..Default::default()
            },
            num_threads: 2,
            debug: false,
            ..Default::default()
        },
        rule_fsts: Some(
            "./matcha-icefall-zh-baker/phone.fst,./matcha-icefall-zh-baker/date.fst,./matcha-icefall-zh-baker/number.fst".into(),
        ),
        ..Default::default()
    };

    let tts = OfflineTts::create(&config).expect("Failed to create OfflineTts");

    println!("Sample rate: {}", tts.sample_rate());
    println!("Num speakers: {}", tts.num_speakers());

    let text = "当夜幕降临，星光点点，伴随着微风拂面，我在静谧中感受着时光的流转，思念如\
        涟漪荡漾，梦境如画卷展开，我与自然融为一体，沉静在这片宁静的美丽之中，感\
        受着生命的奇迹与温柔.\
        某某银行的副行长和一些行政领导表示，他们去过长江和长白山; \
        经济不断增长。2024年12月31号，拨打110或者18920240511。123456块钱。";

    let gen_config = GenerationConfig {
        sid: 0,
        speed: 1.0,
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

    let filename = "./generated-matcha-zh-rust.wav";
    if audio.save(filename) {
        println!("Saved to: {}", filename);
    } else {
        eprintln!("Failed to save {}", filename);
    }
}
