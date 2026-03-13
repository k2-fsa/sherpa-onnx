// Copyright (c)  2026  Xiaomi Corporation
//
// This file demonstrates how to use streaming Zipformer with sherpa-onnx's
// Rust API for real-time streaming speech recognition with a microphone.
//
// See ../README.md for how to run it
//
// See ./streaming_zipformer.rs for how to recognize a wave file.

use anyhow::Result;
use clap::Parser;
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use cpal::SampleFormat;
use sherpa_onnx::{DisplayManager, OnlineRecognizer, OnlineRecognizerConfig};
use std::sync::mpsc;

/// Command-line arguments
#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    #[arg(long)]
    encoder: String,
    #[arg(long)]
    decoder: String,
    #[arg(long)]
    joiner: String,
    #[arg(long)]
    tokens: String,
    #[arg(long, default_value = "cpu")]
    provider: String,
    #[arg(long, default_value_t = false)]
    debug: bool,
    #[arg(long, default_value_t = 3200)]
    chunk_size: usize,
}

/// List input devices and return the default one
fn list_input_devices(host: &cpal::Host) -> Result<cpal::Device> {
    let default_input = host.default_input_device();
    let default_name = default_input.as_ref().map(|d| d.name().unwrap_or_default());

    println!("Available input devices:");
    for device in host.input_devices()? {
        let name = device.name().unwrap_or("<unknown>".to_string());
        let mark = if Some(&name) == default_name.as_ref() {
            "*"
        } else {
            " "
        };
        println!("{} {}", mark, name);
    }

    let device = default_input.ok_or_else(|| anyhow::anyhow!("No default input device"))?;

    println!("\nUsing default device: {}", device.name()?);
    Ok(device)
}

/// Create and configure the OnlineRecognizer
fn setup_recognizer(args: &Args) -> OnlineRecognizer {
    let mut config = OnlineRecognizerConfig::default();
    config.model_config.transducer.encoder = Some(args.encoder.clone());
    config.model_config.transducer.decoder = Some(args.decoder.clone());
    config.model_config.transducer.joiner = Some(args.joiner.clone());
    config.model_config.tokens = Some(args.tokens.clone());
    config.model_config.provider = Some(args.provider.clone());
    config.model_config.debug = args.debug;
    config.enable_endpoint = true;
    config.decoding_method = Some("greedy_search".to_string());

    OnlineRecognizer::create(&config).expect("Failed to create OnlineRecognizer")
}

/// Build the audio input stream (producer)
fn build_input_stream(device: &cpal::Device, tx: mpsc::Sender<Vec<f32>>) -> Result<cpal::Stream> {
    let supported = device.default_input_config()?;
    let config = supported.config();
    let sample_format = supported.sample_format();
    let channels = config.channels as usize;

    let err_fn = |err| eprintln!("Audio stream error: {:?}", err);

    println!(
        "Input format: {:?}, channels: {}, sample_rate: {}",
        sample_format, channels, config.sample_rate.0
    );

    let stream = match sample_format {
        SampleFormat::F32 => device.build_input_stream(
            &config,
            move |data: &[f32], _| {
                if data.is_empty() {
                    return;
                }

                let mono: Vec<f32> = data
                    .chunks(channels)
                    .map(|frame| {
                        let sum: f32 = frame.iter().copied().sum();
                        sum / channels as f32
                    })
                    .collect();
                let _ = tx.send(mono);
            },
            err_fn,
            None,
        )?,

        SampleFormat::I16 => device.build_input_stream(
            &config,
            move |data: &[i16], _| {
                if data.is_empty() {
                    return;
                }

                let mono: Vec<f32> = data
                    .chunks(channels)
                    .map(|frame| {
                        let sum: f32 = frame.iter().map(|&s| s as f32 / i16::MAX as f32).sum();
                        sum / channels as f32
                    })
                    .collect();

                let _ = tx.send(mono);
            },
            err_fn,
            None,
        )?,

        SampleFormat::U16 => device.build_input_stream(
            &config,
            move |data: &[u16], _| {
                if data.is_empty() {
                    return;
                }

                let mono: Vec<f32> = data
                    .chunks(channels)
                    .map(|frame| {
                        let sum: f32 = frame
                            .iter()
                            .map(|&s| {
                                let centered = s as f32 - 32768.0;
                                centered / 32768.0
                            })
                            .sum();
                        sum / channels as f32
                    })
                    .collect();

                let _ = tx.send(mono);
            },
            err_fn,
            None,
        )?,

        other => anyhow::bail!("Unsupported sample format: {:?}", other),
    };

    Ok(stream)
}

/// Main recognition loop (consumer)
fn run_recognition_loop(
    rx: mpsc::Receiver<Vec<f32>>,
    recognizer: &OnlineRecognizer,
    stream: &mut sherpa_onnx::OnlineStream,
    chunk_size: usize,
    sample_rate: i32,
) {
    let mut display = DisplayManager::new();
    let mut buffer = Vec::<f32>::new();

    loop {
        match rx.recv() {
            Ok(samples) => {
                buffer.extend_from_slice(&samples);
            }
            Err(_) => {
                println!("\nAudio stream closed. Exiting.");
                break;
            }
        }

        while buffer.len() >= chunk_size {
            let chunk: Vec<f32> = buffer.drain(..chunk_size).collect();
            stream.accept_waveform(sample_rate, &chunk);

            while recognizer.is_ready(&stream) {
                recognizer.decode(&stream);

                if let Some(result) = recognizer.get_result(&stream) {
                    let text = result.text;
                    if !text.is_empty() {
                        display.update_text(&text);
                    }
                }

                if recognizer.is_endpoint(&stream) {
                    if let Some(result) = recognizer.get_result(&stream) {
                        if !result.text.is_empty() {
                            display.finalize_sentence();
                        }
                    }
                    recognizer.reset(&stream);
                }
            }
        }

        display.render();
    }
}

fn main() -> Result<()> {
    let args = Args::parse();
    let host = cpal::default_host();

    let device = list_input_devices(&host)?;

    let supported = device.default_input_config()?;
    let sample_rate = supported.sample_rate().0 as i32;

    let recognizer = setup_recognizer(&args);
    let mut stream = recognizer.create_stream();

    let (tx, rx) = mpsc::channel::<Vec<f32>>();
    let audio_stream = build_input_stream(&device, tx)?;
    audio_stream.play()?;

    println!("Streaming microphone ASR... Press Ctrl+C to stop.");

    run_recognition_loop(rx, &recognizer, &mut stream, args.chunk_size, sample_rate);

    Ok(())
}
