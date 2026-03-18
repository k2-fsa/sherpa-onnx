// Copyright (c) 2026  zengyw
//
// This file demonstrates how to use online punctuation with sherpa-onnx's Rust API.

use sherpa_onnx::{OnlinePunctuation, OnlinePunctuationConfig, OnlinePunctuationModelConfig};

fn main() {
    let config = OnlinePunctuationConfig {
        model: OnlinePunctuationModelConfig {
            cnn_bilstm: Some("./sherpa-onnx-online-punct-en-2024-08-06/model.onnx".to_string()),
            bpe_vocab: Some("./sherpa-onnx-online-punct-en-2024-08-06/bpe.vocab".to_string()),
            num_threads: 1,
            provider: Some("cpu".to_string()),
            ..Default::default()
        },
    };

    let punct = OnlinePunctuation::create(&config).expect("Failed to create OnlinePunctuation");

    let texts = [
        "how are you i am fine thank you",
        "The African blogosphere is rapidly expanding bringing more voices online in the form of commentaries opinions analyses rants and poetry",
    ];

    println!("----------");
    for text in texts {
        let out = punct
            .add_punctuation(text)
            .expect("Failed to add punctuation");

        println!("Input text: {text}");
        println!("Output text: {out}");
        println!("----------");
    }
}
