// Copyright (c) 2026 Xiaomi Corporation
//
// This file demonstrates how to use sherpa-onnx's Rust API for offline
// punctuation.
//
// See ../README.md for how to run it.

use clap::Parser;
use sherpa_onnx::{OfflinePunctuation, OfflinePunctuationConfig, OfflinePunctuationModelConfig};

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    #[arg(long)]
    ct_transformer: String,

    #[arg(long, default_value_t = 1)]
    num_threads: i32,

    #[arg(long, default_value = "cpu")]
    provider: String,

    #[arg(long, default_value_t = false)]
    debug: bool,
}

fn main() -> anyhow::Result<()> {
    let args = Args::parse();
    let punct = OfflinePunctuation::create(&OfflinePunctuationConfig {
        model: OfflinePunctuationModelConfig {
            ct_transformer: Some(args.ct_transformer),
            num_threads: args.num_threads,
            provider: Some(args.provider),
            debug: args.debug,
        },
    })
    .ok_or_else(|| anyhow::anyhow!("Failed to create OfflinePunctuation"))?;

    let texts = [
        "这是一个测试你好吗How are you我很好thank you are you ok谢谢你",
        "我们都是木头人不会说话不会动",
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
