// Copyright (c)  2026  Xiaomi Corporation
//
// This file demonstrates how to use WeNet CTC with sherpa-onnx's
// Rust API for simulated streaming speech recognition with VAD from a microphone.
//
// It uses Silero VAD to detect speech segments and runs the offline
// WeNet CTC recognizer on each detected segment, providing an experience
// similar to streaming ASR.
//
// See ../README.md for how to run it

use anyhow::Result;
use clap::Parser;
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use cpal::SampleFormat;
use sherpa_onnx::{
    DisplayManager, LinearResampler, OfflineRecognizer, OfflineRecognizerConfig, VadModelConfig,
    VoiceActivityDetector,
};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc;
use std::sync::Arc;
use std::time::Instant;

/// Command-line arguments
#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    #[arg(long)]
    silero_vad_model: String,
    #[arg(long)]
    model: String,
    #[arg(long)]
    tokens: String,
    #[arg(long, default_value_t = 2)]
    num_threads: i32,
    #[arg(long, default_value_t = false)]
    debug: bool,
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

fn create_vad(args: &Args) -> VoiceActivityDetector {
    let mut config = VadModelConfig::default();
    config.silero_vad.model = Some(args.silero_vad_model.clone());
    config.silero_vad.threshold = 0.5;
    config.silero_vad.min_silence_duration = 0.1;
    config.silero_vad.min_speech_duration = 0.25;
    config.silero_vad.max_speech_duration = 8.0;
    config.silero_vad.window_size = 512;
    config.sample_rate = 16000;
    config.debug = args.debug;

    VoiceActivityDetector::create(&config, 20.0).expect("Failed to create VAD")
}

fn create_recognizer(args: &Args) -> OfflineRecognizer {
    let mut config = OfflineRecognizerConfig::default();
    config.model_config.wenet_ctc.model = Some(args.model.clone());
    config.model_config.tokens = Some(args.tokens.clone());
    config.model_config.num_threads = args.num_threads;
    config.model_config.debug = args.debug;

    println!("Loading model");
    let recognizer = OfflineRecognizer::create(&config).expect("Failed to create recognizer");
    println!("Loading model done");
    recognizer
}

fn main() -> Result<()> {
    let args = Args::parse();

    // Set up Ctrl+C handler
    let stop = Arc::new(AtomicBool::new(false));
    let stop_clone = stop.clone();
    ctrlc::set_handler(move || {
        stop_clone.store(true, Ordering::SeqCst);
        eprintln!("\nCaught Ctrl + C. Exiting...");
    })?;

    let vad = create_vad(&args);
    let recognizer = create_recognizer(&args);

    let host = cpal::default_host();
    let device = list_input_devices(&host)?;

    let supported = device.default_input_config()?;
    let mic_sample_rate = supported.sample_rate().0 as i32;
    let resampler = if mic_sample_rate != 16000 {
        Some(LinearResampler::create(mic_sample_rate, 16000).expect("Failed to create resampler"))
    } else {
        None
    };

    let sample_rate = 16000i32;
    let window_size = 512usize; // samples, please don't change

    let (tx, rx) = mpsc::channel::<Vec<f32>>();
    let audio_stream = build_input_stream(&device, tx)?;
    audio_stream.play()?;

    let mut buffer = Vec::<f32>::new();
    let mut offset: usize = 0;
    let mut speech_started = false;
    let mut started_time = Instant::now();
    let mut display = DisplayManager::new();

    println!("Started! Please speak");

    loop {
        if stop.load(Ordering::SeqCst) {
            break;
        }

        match rx.recv() {
            Ok(samples) => {
                if let Some(ref resampler) = resampler {
                    let resampled = resampler.resample(&samples, false);
                    buffer.extend_from_slice(&resampled);
                } else {
                    buffer.extend_from_slice(&samples);
                }
            }
            Err(_) => {
                println!("\nAudio stream closed. Exiting.");
                break;
            }
        }

        // Feed VAD in window_size chunks
        while offset + window_size <= buffer.len() {
            vad.accept_waveform(&buffer[offset..offset + window_size]);
            if !speech_started && vad.detected() {
                speech_started = true;
                started_time = Instant::now();
            }
            offset += window_size;
        }

        // Trim buffer if speech hasn't started and buffer is too large
        if !speech_started && buffer.len() > 10 * window_size {
            let trim_amount = buffer.len() - 10 * window_size;
            offset = offset.saturating_sub(trim_amount);
            buffer = buffer[buffer.len() - 10 * window_size..].to_vec();
        }

        // Interim decode every 0.2 seconds while speech is ongoing
        let elapsed = started_time.elapsed().as_secs_f32();
        if speech_started && elapsed > 0.2 {
            let stream = recognizer.create_stream();
            stream.accept_waveform(sample_rate, &buffer);
            recognizer.decode(&stream);
            if let Some(result) = stream.get_result() {
                display.update_text(&result.text);
                display.render();
            }
            started_time = Instant::now();
        }

        // Process completed VAD segments (final decode)
        while !vad.is_empty() {
            if let Some(segment) = vad.front() {
                vad.pop();

                let stream = recognizer.create_stream();
                stream.accept_waveform(sample_rate, segment.samples());
                recognizer.decode(&stream);
                if let Some(result) = stream.get_result() {
                    display.update_text(&result.text);
                    display.finalize_sentence();
                    display.render();
                }
            }

            buffer.clear();
            offset = 0;
            speech_started = false;
        }

        display.render();
    }

    Ok(())
}
