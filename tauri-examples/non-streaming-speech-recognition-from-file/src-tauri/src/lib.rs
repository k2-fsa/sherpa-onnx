use serde::Serialize;
use sherpa_onnx::{
    LinearResampler, OfflineRecognizer, OfflineRecognizerConfig, OfflineSenseVoiceModelConfig,
    SileroVadModelConfig, VadModelConfig, VoiceActivityDetector,
};
use std::fs::File;
use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};
use std::sync::{Arc, Mutex};
use symphonia::core::audio::SampleBuffer;
use symphonia::core::codecs::CODEC_TYPE_NULL;
use symphonia::core::errors::Error;
use symphonia::core::io::MediaSourceStream;
use symphonia::core::probe::Hint;

#[derive(Serialize, Clone)]
struct SegmentResult {
    start: f32,
    end: f32,
    text: String,
}

#[derive(Serialize, Clone, Default)]
struct ProcessingState {
    percent: u32,
    status: String,
    segments: Vec<SegmentResult>,
}

/// Shared application state, created once at startup.
/// Uses Arc so processing thread can hold references.
struct AppState {
    recognizer: Arc<Mutex<OfflineRecognizer>>,
    vad: Arc<Mutex<VoiceActivityDetector>>,
    cancelled: Arc<AtomicBool>,
    progress: Arc<AtomicU32>,
    status: Arc<Mutex<String>>,
    segments: Arc<Mutex<Vec<SegmentResult>>>,
}

/// Decode an audio or video file to mono f32 PCM samples at the native sample
/// rate.  Supports MP3, FLAC, AAC, OGG, WAV, MP4, MKV, WebM, AIFF, and more
/// via the symphonia crate (pure Rust, no ffmpeg needed).
fn decode_audio_file(path: &str) -> Result<(Vec<f32>, i32), String> {
    let src = File::open(path).map_err(|e| format!("Cannot open file: {e}"))?;
    let mss = MediaSourceStream::new(Box::new(src), Default::default());

    let mut hint = Hint::new();
    if let Some(ext) = std::path::Path::new(path)
        .extension()
        .and_then(|e| e.to_str())
    {
        hint.with_extension(ext);
    }

    let probed = symphonia::default::get_probe()
        .format(&hint, mss, &Default::default(), &Default::default())
        .map_err(|e| format!("Unsupported format: {e}"))?;

    let mut format = probed.format;

    let track = format
        .tracks()
        .iter()
        .find(|t| t.codec_params.codec != CODEC_TYPE_NULL)
        .ok_or_else(|| "No supported audio track found".to_string())?;

    let track_id = track.id;
    let sample_rate = track.codec_params.sample_rate.unwrap_or(16000) as i32;
    let num_channels = track.codec_params.channels.map(|c| c.count()).unwrap_or(1);

    let mut decoder = symphonia::default::get_codecs()
        .make(&track.codec_params, &Default::default())
        .map_err(|e| format!("Cannot create decoder: {e}"))?;

    let mut all_samples: Vec<f32> = Vec::new();

    loop {
        let packet = match format.next_packet() {
            Ok(p) => p,
            Err(Error::IoError(e)) if e.kind() == std::io::ErrorKind::UnexpectedEof => break,
            Err(Error::ResetRequired) => break,
            Err(err) => return Err(format!("Read error: {err}")),
        };

        if packet.track_id() != track_id {
            continue;
        }

        match decoder.decode(&packet) {
            Ok(decoded) => {
                let mut buf = SampleBuffer::<f32>::new(decoded.capacity() as u64, *decoded.spec());
                buf.copy_interleaved_ref(decoded);
                all_samples.extend_from_slice(buf.samples());
            }
            Err(Error::IoError(_)) | Err(Error::DecodeError(_)) => continue,
            Err(err) => return Err(format!("Decode error: {err}")),
        }
    }

    if all_samples.is_empty() {
        return Err("No audio samples decoded".to_string());
    }

    // Mix multi-channel to mono by averaging channels
    let mono = if num_channels > 1 {
        let nc = num_channels;
        let num_frames = all_samples.len() / nc;
        let mut mono = Vec::with_capacity(num_frames);
        for frame in 0..num_frames {
            let mut sum = 0.0f32;
            for ch in 0..nc {
                sum += all_samples[frame * nc + ch];
            }
            mono.push(sum / nc as f32);
        }
        mono
    } else {
        all_samples
    };

    Ok((mono, sample_rate))
}

/// Start recognition in a background thread. Returns immediately.
/// The frontend should poll `get_recognition_progress` to track progress.
#[tauri::command]
fn recognize_file(path: String, state: tauri::State<'_, AppState>) -> Result<(), String> {
    // Reset state
    state.cancelled.store(false, Ordering::Relaxed);
    state.progress.store(0, Ordering::Relaxed);
    *state.status.lock().map_err(|e| e.to_string())? = "processing".to_string();
    state.segments.lock().map_err(|e| e.to_string())?.clear();

    // Clone Arc handles for the worker thread
    let recognizer = Arc::clone(&state.recognizer);
    let vad = Arc::clone(&state.vad);
    let cancelled = Arc::clone(&state.cancelled);
    let progress = Arc::clone(&state.progress);
    let status = Arc::clone(&state.status);
    let segments = Arc::clone(&state.segments);

    std::thread::spawn(move || {
        let result = run_recognition(&path, &recognizer, &vad, &cancelled, &progress, &segments);

        let mut s = status.lock().unwrap();
        match result {
            Ok(()) => {
                if cancelled.load(Ordering::Relaxed) {
                    *s = "cancelled".to_string();
                } else {
                    progress.store(100, Ordering::Relaxed);
                    *s = "done".to_string();
                }
            }
            Err(e) => {
                *s = format!("error: {e}");
            }
        }
    });

    Ok(())
}

fn run_recognition(
    path: &str,
    recognizer: &Arc<Mutex<OfflineRecognizer>>,
    vad: &Arc<Mutex<VoiceActivityDetector>>,
    cancelled: &AtomicBool,
    progress: &AtomicU32,
    segments: &Arc<Mutex<Vec<SegmentResult>>>,
) -> Result<(), String> {
    let (samples, sample_rate) = decode_audio_file(path)?;

    let resampled;
    let input_samples = if sample_rate != 16000 {
        let resampler = LinearResampler::create(sample_rate, 16000)
            .ok_or_else(|| format!("Failed to create resampler for {sample_rate} Hz"))?;
        resampled = resampler.resample(&samples, true);
        resampled.as_slice()
    } else {
        samples.as_slice()
    };

    let total_samples = input_samples.len();

    let recognizer = recognizer.lock().map_err(|e| e.to_string())?;
    let vad = vad.lock().map_err(|e| e.to_string())?;
    vad.reset();

    let window_size: usize = 512;
    let mut last_progress: u32 = 0;

    let mut i = 0;
    while i < total_samples {
        if cancelled.load(Ordering::Relaxed) {
            vad.clear();
            break;
        }

        let end = (i + window_size).min(total_samples);
        if end - i == window_size {
            vad.accept_waveform(&input_samples[i..end]);
        } else {
            vad.flush();
        }

        while let Some(segment) = vad.front() {
            let seg_samples = segment.samples();
            let duration = seg_samples.len() as f32 / 16000.0;
            let start_time = segment.start() as f32 / 16000.0;
            let end_time = start_time + duration;

            if duration >= 0.1 {
                let stream = recognizer.create_stream();
                stream.accept_waveform(16000, seg_samples);
                recognizer.decode(&stream);
                if let Some(r) = stream.get_result() {
                    if !r.text.is_empty() {
                        if let Ok(mut segs) = segments.lock() {
                            segs.push(SegmentResult {
                                start: start_time,
                                end: end_time,
                                text: r.text,
                            });
                        }
                    }
                }
            }
            vad.pop();
        }

        i += window_size;

        if i % 20480 < window_size || i >= total_samples {
            let percent = ((i as f32 / total_samples as f32) * 100.0) as u32;
            let percent = percent.min(100);
            if percent != last_progress {
                last_progress = percent;
                progress.store(percent, Ordering::Relaxed);
            }
        }
    }

    Ok(())
}

/// Poll this from the frontend to get current progress and results.
#[tauri::command]
fn get_recognition_progress(state: tauri::State<'_, AppState>) -> Result<ProcessingState, String> {
    let percent = state.progress.load(Ordering::Relaxed);
    let status = state.status.lock().map_err(|e| e.to_string())?.clone();
    let segments = state.segments.lock().map_err(|e| e.to_string())?.clone();
    Ok(ProcessingState {
        percent,
        status,
        segments,
    })
}

/// Cancel the current recognition.
#[tauri::command]
fn cancel_recognition(state: tauri::State<'_, AppState>) {
    state.cancelled.store(true, Ordering::Relaxed);
}

fn format_srt_time(seconds: f32) -> String {
    let total_ms = (seconds * 1000.0) as u64;
    let h = total_ms / 3_600_000;
    let m = (total_ms % 3_600_000) / 60_000;
    let s = (total_ms % 60_000) / 1000;
    let ms = total_ms % 1000;
    format!("{h:02}:{m:02}:{s:02},{ms:03}")
}

/// Export the current results as an SRT subtitle file.
#[tauri::command]
fn export_srt(path: String, state: tauri::State<'_, AppState>) -> Result<(), String> {
    let segments = state.segments.lock().map_err(|e| e.to_string())?;
    if segments.is_empty() {
        return Err("No results to export".to_string());
    }

    let mut srt = String::new();
    for (i, seg) in segments.iter().enumerate() {
        srt.push_str(&format!("{}\n", i + 1));
        srt.push_str(&format!(
            "{} --> {}\n",
            format_srt_time(seg.start),
            format_srt_time(seg.end)
        ));
        srt.push_str(&seg.text);
        srt.push_str("\n\n");
    }

    std::fs::write(&path, srt).map_err(|e| format!("Cannot write file: {e}"))?;
    Ok(())
}

/// Create the recognizer and VAD once at startup.
fn build_app_state() -> AppState {
    let tokens = "./sherpa-onnx-sense-voice-zh-en-ja-ko-yue-int8-2024-07-17/tokens.txt";
    let sense_voice_model =
        "./sherpa-onnx-sense-voice-zh-en-ja-ko-yue-int8-2024-07-17/model.int8.onnx";
    let silero_vad_model = "./silero_vad.onnx";

    let mut asr_config = OfflineRecognizerConfig::default();
    asr_config.model_config.sense_voice = OfflineSenseVoiceModelConfig {
        model: Some(sense_voice_model.to_string()),
        language: Some("auto".to_string()),
        use_itn: true,
        ..Default::default()
    };
    asr_config.model_config.tokens = Some(tokens.to_string());
    asr_config.model_config.num_threads = 2;

    let recognizer = OfflineRecognizer::create(&asr_config)
        .expect("Failed to create recognizer. Check model paths.");

    let mut vad_config = VadModelConfig::default();
    vad_config.silero_vad = SileroVadModelConfig {
        model: Some(silero_vad_model.to_string()),
        threshold: 0.5,
        min_silence_duration: 0.1,
        min_speech_duration: 0.25,
        window_size: 512,
        max_speech_duration: 8.0,
        ..Default::default()
    };
    vad_config.sample_rate = 16000;
    vad_config.num_threads = 1;

    let vad = VoiceActivityDetector::create(&vad_config, 120.0)
        .expect("Failed to create VAD. Check silero_vad model path.");

    AppState {
        recognizer: Arc::new(Mutex::new(recognizer)),
        vad: Arc::new(Mutex::new(vad)),
        cancelled: Arc::new(AtomicBool::new(false)),
        progress: Arc::new(AtomicU32::new(0)),
        status: Arc::new(Mutex::new("idle".to_string())),
        segments: Arc::new(Mutex::new(Vec::new())),
    }
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    let state = build_app_state();

    tauri::Builder::default()
        .plugin(tauri_plugin_dialog::init())
        .manage(state)
        .invoke_handler(tauri::generate_handler![
            recognize_file,
            get_recognition_progress,
            cancel_recognition,
            export_srt,
        ])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
