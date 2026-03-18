// Copyright (c) 2026 zengyw
//
// This file demonstrates how to use online punctuation with sherpa-onnx's Rust API.
//
// See ../README.md for how to run it.

use clap::Parser;
use sherpa_onnx::{OnlinePunctuation, OnlinePunctuationConfig, OnlinePunctuationModelConfig};

/// Online punctuation example
#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Path to CNN-BiLSTM ONNX model
    #[arg(long)]
    cnn_bilstm: String,

    /// Path to BPE vocabulary file
    #[arg(long)]
    bpe_vocab: String,

    /// Number of threads
    #[arg(long, default_value_t = 1)]
    num_threads: i32,

    /// Provider (default: cpu)
    #[arg(long, default_value = "cpu")]
    provider: String,

    /// Enable debug logs
    #[arg(long, default_value_t = false)]
    debug: bool,
}

fn main() -> anyhow::Result<()> {
    let args = Args::parse();

    let config = OnlinePunctuationConfig {
        model: OnlinePunctuationModelConfig {
            cnn_bilstm: Some(args.cnn_bilstm),
            bpe_vocab: Some(args.bpe_vocab),
            num_threads: args.num_threads,
            provider: Some(args.provider),
            debug: args.debug,
            ..Default::default()
        },
    };

    let punct = OnlinePunctuation::create(&config)
        .ok_or_else(|| anyhow::anyhow!("Failed to create OnlinePunctuation"))?;

    let texts = [
        "how are you i am fine thank you",
        "The African blogosphere is rapidly expanding bringing more voices online in the form of commentaries opinions analyses rants and poetry",
    ];

    println!("----------");
    for text in texts {
        let out = punct
            .add_punctuation(text)
            .ok_or_else(|| anyhow::anyhow!("Failed to add punctuation"))?;

        println!("Input text: {text}");
        println!("Output text: {out}");
        println!("----------");
    }

    Ok(())
}
