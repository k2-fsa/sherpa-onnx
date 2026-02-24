// Copyright (c)  2026  Xiaomi Corporation
//
// This file demonstrates how to use streaming Zipformer with sherpa-onnx's
// Rust API for speech recognition.
//
// See ../README.md for how to run it
//
// Note that even if we use a wave file as an example, this model supports
// real-time streaming speech recognition. You can read audio samples
// from a microphone.

use sherpa_onnx::{OnlineRecognizer, OnlineRecognizerConfig, Wave};

fn main() {
    let wav_path = "sherpa-onnx-streaming-zipformer-en-2023-06-21/test_wavs/1.wav";
    let encoder_path =
        "sherpa-onnx-streaming-zipformer-en-2023-06-21/encoder-epoch-99-avg-1.int8.onnx";
    let decoder_path = "sherpa-onnx-streaming-zipformer-en-2023-06-21/decoder-epoch-99-avg-1.onnx";
    let joiner_path =
        "sherpa-onnx-streaming-zipformer-en-2023-06-21/joiner-epoch-99-avg-1.int8.onnx";
    let tokens_path = "sherpa-onnx-streaming-zipformer-en-2023-06-21/tokens.txt";
    let provider = "cpu";

    let wave = Wave::read(wav_path).expect("Failed to read WAV file");

    let mut recognizer_config = OnlineRecognizerConfig::default();
    recognizer_config.model_config.transducer.encoder = Some(encoder_path.to_string());
    recognizer_config.model_config.transducer.decoder = Some(decoder_path.to_string());
    recognizer_config.model_config.transducer.joiner = Some(joiner_path.to_string());
    recognizer_config.model_config.tokens = Some(tokens_path.to_string());
    recognizer_config.model_config.provider = Some(provider.to_string());
    recognizer_config.enable_endpoint = true;

    // set to true to see verbose logs
    recognizer_config.model_config.debug = true;

    recognizer_config.decoding_method = Some("greedy_search".to_string());

    let recognizer =
        OnlineRecognizer::create(&recognizer_config).expect("Failed to create OnlineRecognizer");

    let stream = recognizer.create_stream();

    let mut segment_id = 0;

    // use an use any postive value you like
    const N: usize = 3200; // chunk size for streaming

    println!(
        "Sample rate: {}, num samples: {}, duration: {:.2}s",
        wave.sample_rate(),
        wave.num_samples(),
        wave.num_samples() as f32 / wave.sample_rate() as f32
    );

    let samples = wave.samples();
    let mut k = 0;
    while k < samples.len() {
        let end = (k + N).min(samples.len());
        let chunk = &samples[k..end];
        k = end;

        stream.accept_waveform(wave.sample_rate(), chunk);

        while recognizer.is_ready(&stream) {
            recognizer.decode(&stream);

            if let Some(result) = recognizer.get_result(&stream) {
                if !result.text.is_empty() {
                    println!("Segment {}: {}", segment_id, result.text);
                }
            }

            if recognizer.is_endpoint(&stream) {
                recognizer.reset(&stream);
                segment_id += 1;
            }
        }
    }

    // Tail padding (0.3s)
    let tail_padding = vec![0.0f32; 4800];
    stream.accept_waveform(wave.sample_rate(), &tail_padding);

    stream.input_finished();

    while recognizer.is_ready(&stream) {
        recognizer.decode(&stream);
        if let Some(result) = recognizer.get_result(&stream) {
            if !result.text.is_empty() {
                println!("Segment {}: {}", segment_id, result.text);
            }
        }
    }

    println!("Transcription finished.");
}
