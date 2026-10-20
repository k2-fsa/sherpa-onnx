// rust-api-examples/examples/streaming_zipformer_ctc_hlg.rs
//
// Copyright (c)  2026  Xiaomi Corporation
//
// This file demonstrates how to decode a streaming CTC model with an HLG
// decoding graph using sherpa-onnx's Rust API.
//
// The HLG graph is passed to the C API through ctc_fst_decoder_config.
//
// clang-format off
//
// wget https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-streaming-zipformer-ctc-small-2024-03-18.tar.bz2
// tar xvf sherpa-onnx-streaming-zipformer-ctc-small-2024-03-18.tar.bz2
// rm sherpa-onnx-streaming-zipformer-ctc-small-2024-03-18.tar.bz2
//
// cargo run --release --example streaming_zipformer_ctc_hlg -- \
//   --model ./sherpa-onnx-streaming-zipformer-ctc-small-2024-03-18/ctc-epoch-30-avg-3-chunk-16-left-128.int8.onnx \
//   --tokens ./sherpa-onnx-streaming-zipformer-ctc-small-2024-03-18/tokens.txt \
//   --graph ./sherpa-onnx-streaming-zipformer-ctc-small-2024-03-18/HLG.fst \
//   --wav ./sherpa-onnx-streaming-zipformer-ctc-small-2024-03-18/test_wavs/8k.wav
//
// clang-format on

use clap::Parser;
use sherpa_onnx::{OnlineCtcFstDecoderConfig, OnlineRecognizer, OnlineRecognizerConfig, Wave};

/// Streaming CTC decoding with an HLG graph via ctc_fst_decoder_config.
#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Path to the streaming zipformer CTC model
    #[arg(long)]
    model: String,

    /// Path to tokens.txt
    #[arg(long)]
    tokens: String,

    /// Path to the decoding graph (H.fst, HL.fst or HLG.fst)
    #[arg(long)]
    graph: String,

    /// Path to a wav file
    #[arg(long)]
    wav: String,

    /// Number of threads
    #[arg(long, default_value_t = 1)]
    num_threads: i32,

    /// Enable debug logs
    #[arg(long, default_value_t = false)]
    debug: bool,
}

fn main() {
    let args = Args::parse();

    let wave = Wave::read(&args.wav).expect("Failed to read WAV file");

    let mut config = OnlineRecognizerConfig::default();
    config.model_config.zipformer2_ctc.model = Some(args.model);
    config.model_config.tokens = Some(args.tokens);
    config.model_config.num_threads = args.num_threads;
    config.model_config.debug = args.debug;

    // This is what the example exercises: the graph and max_active reach the C
    // API through OnlineCtcFstDecoderConfig.
    config.ctc_fst_decoder_config = OnlineCtcFstDecoderConfig {
        graph: Some(args.graph),
        max_active: 3000,
    };

    let recognizer =
        OnlineRecognizer::create(&config).expect("Failed to create OnlineRecognizer");

    let stream = recognizer.create_stream();
    let mut segment_id = 0;

    println!(
        "Sample rate: {}, num samples: {}, duration: {:.2}s",
        wave.sample_rate(),
        wave.num_samples(),
        wave.num_samples() as f32 / wave.sample_rate() as f32
    );

    // simulate streaming. You can choose an arbitrary N
    const CHUNK_SIZE: usize = 3200;

    for chunk in wave.samples().chunks(CHUNK_SIZE) {
        stream.accept_waveform(wave.sample_rate(), chunk);

        while recognizer.is_ready(&stream) {
            recognizer.decode(&stream);
        }

        if let Some(result) = recognizer.get_result(&stream) {
            if !result.text.is_empty() {
                println!("Segment {}: {}", segment_id, result.text);
            }
        }

        if recognizer.is_endpoint(&stream) {
            if let Some(result) = recognizer.get_result(&stream) {
                if !result.text.is_empty() {
                    segment_id += 1;
                }
            }
            recognizer.reset(&stream);
        }
    }

    // Tail padding (~0.3s)
    let tail_padding_len = (wave.sample_rate() as f32 * 0.3).round() as usize;
    let tail_padding = vec![0.0f32; tail_padding_len];
    stream.accept_waveform(wave.sample_rate(), &tail_padding);

    stream.input_finished();

    while recognizer.is_ready(&stream) {
        recognizer.decode(&stream);
    }

    if let Some(result) = recognizer.get_result(&stream) {
        if !result.text.is_empty() {
            println!("Segment {}: {}", segment_id, result.text);
        }
    }

    println!("Transcription finished.");
}
