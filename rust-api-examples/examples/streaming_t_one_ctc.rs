// This file demonstrates how to use streaming T-one CTC with sherpa-onnx's
// Rust API for speech recognition (Russian, 8 kHz).
//
// Run it from the rust-api-examples directory with:
//   ./run-streaming-t-one-ctc.sh
//
// Note that even if we use a wave file as an example, this model supports
// real-time streaming speech recognition.
//
// See also ../../c-api-examples/streaming-t-one-ctc-c-api.c

use clap::Parser;
use sherpa_onnx::{OnlineRecognizer, OnlineRecognizerConfig, Wave};

/// Simple streaming T-one CTC example
#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Path to WAV file
    #[arg(long)]
    wav: String,

    /// Path to the T-one CTC ONNX model
    #[arg(long)]
    model: String,

    /// Path to tokens file
    #[arg(long)]
    tokens: String,

    /// Provider (default: cpu)
    #[arg(long, default_value = "cpu")]
    provider: String,

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

    let mut recognizer_config = OnlineRecognizerConfig::default();
    recognizer_config.model_config.t_one_ctc.model = Some(args.model.clone());
    recognizer_config.model_config.tokens = Some(args.tokens.clone());
    recognizer_config.model_config.provider = Some(args.provider.clone());
    recognizer_config.model_config.num_threads = args.num_threads;
    recognizer_config.model_config.debug = args.debug;
    // Like the Python example, decode the whole file as one utterance.
    // See ./streaming_zipformer.rs for how to use endpointing.
    recognizer_config.enable_endpoint = false;
    recognizer_config.decoding_method = Some("greedy_search".to_string());

    let recognizer =
        OnlineRecognizer::create(&recognizer_config).expect("Failed to create OnlineRecognizer");

    let stream = recognizer.create_stream();

    // use any positive value as you like
    const CHUNK_SIZE: usize = 3200;

    println!(
        "Sample rate: {}, num samples: {}, duration: {:.2}s",
        wave.sample_rate(),
        wave.num_samples(),
        wave.num_samples() as f32 / wave.sample_rate() as f32
    );

    // T-one expects some silence before the audio (~0.3s)
    let left_padding_len = (wave.sample_rate() as f32 * 0.3).round() as usize;
    let left_padding = vec![0.0f32; left_padding_len];
    stream.accept_waveform(wave.sample_rate(), &left_padding);

    // Feed the audio in small chunks to simulate streaming
    for chunk in wave.samples().chunks(CHUNK_SIZE) {
        stream.accept_waveform(wave.sample_rate(), chunk);

        while recognizer.is_ready(&stream) {
            recognizer.decode(&stream);
        }

        // Print the partial result so far
        if let Some(result) = recognizer.get_result(&stream) {
            if !result.text.is_empty() {
                println!("Partial: {}", result.text);
            }
        }
    }

    // Tail padding (~0.6s) so the last words get decoded
    let tail_padding_len = (wave.sample_rate() as f32 * 0.6).round() as usize;
    let tail_padding = vec![0.0f32; tail_padding_len];
    stream.accept_waveform(wave.sample_rate(), &tail_padding);

    stream.input_finished();

    while recognizer.is_ready(&stream) {
        recognizer.decode(&stream);
    }

    if let Some(result) = recognizer.get_result(&stream) {
        println!("Final: {}", result.text);
    }

    println!("Transcription finished.");
}
