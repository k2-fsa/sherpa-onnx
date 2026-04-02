// Copyright (c) 2026 Xiaomi Corporation
//
// This file demonstrates how to use sherpa-onnx's Rust API for keyword spotting.
//
// See ../README.md for how to run it.

use clap::Parser;
use sherpa_onnx::{KeywordSpotter, KeywordSpotterConfig, Wave};

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    #[arg(long)]
    wav: String,

    #[arg(long)]
    encoder: String,

    #[arg(long)]
    decoder: String,

    #[arg(long)]
    joiner: String,

    #[arg(long)]
    tokens: String,

    #[arg(long)]
    keywords_file: String,

    #[arg(long, default_value = "cpu")]
    provider: String,

    #[arg(long, default_value_t = 1)]
    num_threads: i32,

    #[arg(long, default_value_t = false)]
    debug: bool,
}

fn detect_keywords(
    kws: &KeywordSpotter,
    wave: &Wave,
    title: &str,
    extra_keywords: Option<&str>,
) {
    println!("{title}");

    let stream = if let Some(extra_keywords) = extra_keywords {
        kws.create_stream_with_keywords(extra_keywords)
    } else {
        kws.create_stream()
    };

    let tail_padding = vec![0.0f32; (wave.sample_rate() / 2) as usize];
    stream.accept_waveform(wave.sample_rate(), wave.samples());
    stream.accept_waveform(wave.sample_rate(), &tail_padding);
    stream.input_finished();

    let mut detected = false;
    while kws.is_ready(&stream) {
        kws.decode(&stream);
        if let Some(result) = kws.get_result(&stream) {
            if !result.keyword.is_empty() {
                detected = true;
                println!("Detected keyword: {}", result.json);
                kws.reset(&stream);
            }
        }
    }

    if !detected {
        println!("No keyword detected.");
    }

    println!();
}

fn main() -> anyhow::Result<()> {
    let args = Args::parse();
    let wave = Wave::read(&args.wav).ok_or_else(|| anyhow::anyhow!("Failed to read WAV file"))?;

    let mut config = KeywordSpotterConfig::default();
    config.model_config.transducer.encoder = Some(args.encoder);
    config.model_config.transducer.decoder = Some(args.decoder);
    config.model_config.transducer.joiner = Some(args.joiner);
    config.model_config.tokens = Some(args.tokens);
    config.model_config.provider = Some(args.provider);
    config.model_config.num_threads = args.num_threads;
    config.model_config.debug = args.debug;
    config.keywords_file = Some(args.keywords_file);

    let kws = KeywordSpotter::create(&config)
        .ok_or_else(|| anyhow::anyhow!("Failed to create KeywordSpotter"))?;

    detect_keywords(
        &kws,
        &wave,
        "--Test pre-defined keywords from the keywords file--",
        None,
    );
    detect_keywords(
        &kws,
        &wave,
        "--Use pre-defined keywords + add a new keyword--",
        Some("y ǎn y uán @演员"),
    );
    detect_keywords(
        &kws,
        &wave,
        "--Use pre-defined keywords + add two new keywords--",
        Some("y ǎn y uán @演员/zh ī m íng @知名"),
    );

    Ok(())
}
