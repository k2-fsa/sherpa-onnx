mod model_registry;

use model_registry::get_model_config;
use serde::Serialize;
use sherpa_onnx::{
    LinearResampler, OfflineRecognizer, SileroVadModelConfig, VadModelConfig,
    VoiceActivityDetector,
};
use std::fs::File;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU8, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Instant;
use symphonia::core::audio::{AudioBufferRef, SampleBuffer};
use symphonia::core::codecs::{Decoder, CODEC_TYPE_NULL};
use symphonia::core::formats::FormatReader;
use symphonia::core::io::MediaSourceStream;
use symphonia::core::probe::Hint;

/// Which ASR model to bundle. Build scripts patch these via sed.
const MODEL_TYPE: u32 = 15;
const MODEL_NAME: &str = "sherpa-onnx-sense-voice-zh-en-ja-ko-yue-int8-2024-07-17";

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
    elapsed_secs: f64,
    audio_duration_secs: f64,
}

/// Shared application state, created once at startup.
/// Uses Arc so processing thread can hold references.
/// Recognizer and VAD are Option because they are initialized in a background thread.
/// init_status: 0 = pending, 1 = ready, 2 = error.
struct AppState {
    recognizer: Arc<Mutex<Option<OfflineRecognizer>>>,
    vad: Arc<Mutex<Option<VoiceActivityDetector>>>,
    cancelled: Arc<AtomicBool>,
    progress: Arc<AtomicU32>,
    status: Arc<Mutex<String>>,
    segments: Arc<Mutex<Vec<SegmentResult>>>,
    audio_path: Arc<Mutex<String>>,
    init_status: Arc<AtomicU8>,
    init_error: Arc<Mutex<String>>,
    elapsed_ms: Arc<AtomicU32>,
    audio_duration_ms: Arc<AtomicU32>,
    num_threads: Arc<AtomicU32>,
}

/// Open an audio/video file and return the format reader, decoder, and track info.
/// Supports MP3, FLAC, AAC, OGG, WAV, MP4, MKV, WebM, AIFF, and more via symphonia.
fn open_audio_file(
    path: &str,
) -> Result<(Box<dyn FormatReader>, Box<dyn Decoder>, u32, usize, u32), String> {
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

    let format = probed.format;

    let track = format
        .tracks()
        .iter()
        .find(|t| t.codec_params.codec != CODEC_TYPE_NULL)
        .ok_or_else(|| "No supported audio track found".to_string())?;

    let track_id = track.id;
    let sample_rate = track.codec_params.sample_rate.unwrap_or(16000);
    let num_channels = track
        .codec_params
        .channels
        .map(|c| c.count())
        .unwrap_or(1);

    let decoder = symphonia::default::get_codecs()
        .make(&track.codec_params, &Default::default())
        .map_err(|e| format!("Cannot create decoder: {e}"))?;

    Ok((format, decoder, track_id, num_channels, sample_rate))
}

/// Decode interleaved AudioBufferRef to mono f32 samples.
fn decode_to_mono_f32(decoded: &AudioBufferRef, num_channels: usize) -> Vec<f32> {
    let mut buf =
        SampleBuffer::<f32>::new(decoded.capacity() as u64, *decoded.spec());
    buf.copy_interleaved_ref(decoded.clone());
    let all = buf.samples();

    if num_channels == 1 {
        return all.to_vec();
    }

    let nc = num_channels;
    let num_frames = all.len() / nc;
    let mut mono = Vec::with_capacity(num_frames);
    for frame in 0..num_frames {
        let mut sum = 0.0f32;
        for ch in 0..nc {
            sum += all[frame * nc + ch];
        }
        mono.push(sum / nc as f32);
    }
    mono
}

/// Start recognition in a background thread. Returns immediately.
/// The frontend should poll `get_recognition_progress` to track progress.
/// Streams audio packet-by-packet to avoid loading the entire file into memory.
#[tauri::command]
fn recognize_file(path: String, state: tauri::State<'_, AppState>) -> Result<(), String> {
    // Check initialization
    let init = state.init_status.load(Ordering::Relaxed);
    if init == 0 {
        return Err("Models are still loading, please wait".to_string());
    }
    if init == 2 {
        let err = state.init_error.lock().map_err(|e| e.to_string())?.clone();
        return Err(format!("Initialization failed: {err}"));
    }

    // Reset state
    state.cancelled.store(false, Ordering::Relaxed);
    state.progress.store(0, Ordering::Relaxed);
    state.elapsed_ms.store(0, Ordering::Relaxed);
    state.audio_duration_ms.store(0, Ordering::Relaxed);
    *state.status.lock().map_err(|e| e.to_string())? = "processing".to_string();
    state.segments.lock().map_err(|e| e.to_string())?.clear();
    *state.audio_path.lock().map_err(|e| e.to_string())? = path.clone();

    // Clone Arc handles for the worker thread
    let recognizer = Arc::clone(&state.recognizer);
    let vad = Arc::clone(&state.vad);
    let elapsed_ms = Arc::clone(&state.elapsed_ms);
    let audio_duration_ms = Arc::clone(&state.audio_duration_ms);
    let cancelled = Arc::clone(&state.cancelled);
    let progress = Arc::clone(&state.progress);
    let status = Arc::clone(&state.status);
    let segments = Arc::clone(&state.segments);

    std::thread::spawn(move || {
        let start = Instant::now();
        let result = run_recognition(
            &path,
            &recognizer,
            &vad,
            &cancelled,
            &progress,
            &segments,
            &audio_duration_ms,
            &elapsed_ms,
            &start,
        );

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

/// Stream audio through VAD + ASR without buffering the entire file.
fn run_recognition(
    path: &str,
    recognizer: &Arc<Mutex<Option<OfflineRecognizer>>>,
    vad: &Arc<Mutex<Option<VoiceActivityDetector>>>,
    cancelled: &AtomicBool,
    progress: &Arc<AtomicU32>,
    segments: &Arc<Mutex<Vec<SegmentResult>>>,
    audio_duration_ms: &Arc<AtomicU32>,
    elapsed_ms: &Arc<AtomicU32>,
    start: &Instant,
) -> Result<(), String> {
    let (mut format_reader, mut decoder, track_id, num_channels, native_rate) =
        open_audio_file(path)?;

    let resampler = if native_rate != 16000 {
        Some(
            LinearResampler::create(native_rate as i32, 16000)
                .ok_or_else(|| format!("Failed to create resampler for {native_rate} Hz"))?,
        )
    } else {
        None
    };

    // Get approximate total samples for progress reporting.
    // If not available, progress will stay at 0% until completion.
    let total_samples: Option<usize> = format_reader
        .default_track()
        .and_then(|t| t.codec_params.n_frames)
        .map(|n| n as usize);

    // Compute audio duration for RTF display
    if let Some(total) = total_samples {
        let sr = format_reader
            .default_track()
            .and_then(|t| t.codec_params.sample_rate)
            .unwrap_or(16000) as usize;
        if sr > 0 {
            let dur_ms = ((total as f64 / sr as f64) * 1000.0) as u32;
            audio_duration_ms.store(dur_ms, Ordering::Relaxed);
        }
    }

    let mut recognizer_guard = recognizer.lock().map_err(|e| e.to_string())?;
    let recognizer = recognizer_guard.as_mut().ok_or("Recognizer not initialized")?;
    let mut vad_guard = vad.lock().map_err(|e| e.to_string())?;
    let vad = vad_guard.as_mut().ok_or("VAD not initialized")?;
    vad.reset();

    let window_size: usize = 512;
    let mut vad_buf: Vec<f32> = Vec::new();
    let mut decoded_count: usize = 0;
    let mut last_progress: u32 = 0;
    let mut last_elapsed_update = Instant::now();

    loop {
        if cancelled.load(Ordering::Relaxed) {
            vad.clear();
            break;
        }

        let packet = match format_reader.next_packet() {
            Ok(p) => p,
            Err(symphonia::core::errors::Error::IoError(e))
                if e.kind() == std::io::ErrorKind::UnexpectedEof =>
            {
                break;
            }
            Err(symphonia::core::errors::Error::ResetRequired) => break,
            Err(err) => return Err(format!("Read error: {err}")),
        };

        if packet.track_id() != track_id {
            continue;
        }

        let decoded = match decoder.decode(&packet) {
            Ok(d) => d,
            Err(symphonia::core::errors::Error::IoError(_))
            | Err(symphonia::core::errors::Error::DecodeError(_)) => continue,
            Err(err) => return Err(format!("Decode error: {err}")),
        };

        let mono = decode_to_mono_f32(&decoded, num_channels);
        decoded_count += mono.len();

        // Resample to 16 kHz if needed
        let pcm = if let Some(ref resamp) = resampler {
            resamp.resample(&mono, false)
        } else {
            mono
        };

        vad_buf.extend_from_slice(&pcm);

        // Feed 512-sample windows to VAD
        while vad_buf.len() >= window_size {
            if cancelled.load(Ordering::Relaxed) {
                vad.clear();
                return Ok(());
            }

            vad.accept_waveform(&vad_buf[..window_size]);
            vad_buf.drain(..window_size);

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
        }

        // Report progress
        if let Some(total) = total_samples {
            if total > 0 {
                let percent = ((decoded_count as f32 / total as f32) * 100.0) as u32;
                let percent = percent.min(99); // 100% is set on completion
                if percent != last_progress {
                    last_progress = percent;
                    progress.store(percent, Ordering::Relaxed);
                }
            }
        }

        // Update elapsed time periodically (at least every 500ms)
        if last_elapsed_update.elapsed().as_millis() >= 500 {
            elapsed_ms.store(start.elapsed().as_millis() as u32, Ordering::Relaxed);
            last_elapsed_update = Instant::now();
        }
    }

    // Flush remaining samples through VAD
    if !cancelled.load(Ordering::Relaxed) && !vad_buf.is_empty() {
        vad.flush();
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
    }

    Ok(())
}

/// Poll this from the frontend to get current progress and results.
#[tauri::command]
fn get_recognition_progress(state: tauri::State<'_, AppState>) -> Result<ProcessingState, String> {
    let percent = state.progress.load(Ordering::Relaxed);
    let status = state.status.lock().map_err(|e| e.to_string())?.clone();
    let segments = state.segments.lock().map_err(|e| e.to_string())?.clone();
    let elapsed_ms = state.elapsed_ms.load(Ordering::Relaxed);
    let audio_dur_ms = state.audio_duration_ms.load(Ordering::Relaxed);
    let elapsed_secs = elapsed_ms as f64 / 1000.0;
    let audio_duration_secs = audio_dur_ms as f64 / 1000.0;
    Ok(ProcessingState {
        percent,
        status,
        segments,
        elapsed_secs,
        audio_duration_secs,
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

/// Write mono f32 PCM samples as a 16-bit WAV file at 16 kHz.
fn write_wav(path: &str, samples: &[f32]) -> Result<(), String> {
    let num_samples = samples.len() as u32;
    let byte_rate = 16000u32 * 2;
    let data_size = num_samples * 2;
    let file_size = 36 + data_size;

    let mut f = File::create(path).map_err(|e| format!("Cannot create file: {e}"))?;

    use std::io::Write;
    f.write_all(b"RIFF").map_err(|e| e.to_string())?;
    f.write_all(&file_size.to_le_bytes()).map_err(|e| e.to_string())?;
    f.write_all(b"WAVE").map_err(|e| e.to_string())?;
    f.write_all(b"fmt ").map_err(|e| e.to_string())?;
    f.write_all(&16u32.to_le_bytes()).map_err(|e| e.to_string())?;
    f.write_all(&1u16.to_le_bytes()).map_err(|e| e.to_string())?;
    f.write_all(&1u16.to_le_bytes()).map_err(|e| e.to_string())?;
    f.write_all(&16000u32.to_le_bytes()).map_err(|e| e.to_string())?;
    f.write_all(&byte_rate.to_le_bytes()).map_err(|e| e.to_string())?;
    f.write_all(&2u16.to_le_bytes()).map_err(|e| e.to_string())?;
    f.write_all(&16u16.to_le_bytes()).map_err(|e| e.to_string())?;
    f.write_all(b"data").map_err(|e| e.to_string())?;
    f.write_all(&data_size.to_le_bytes()).map_err(|e| e.to_string())?;
    for &s in samples {
        let clamped = s.max(-1.0).min(1.0);
        let pcm = (clamped * 32767.0) as i16;
        f.write_all(&pcm.to_le_bytes()).map_err(|e| e.to_string())?;
    }

    Ok(())
}

/// Decode audio from the file for a specific time range (in seconds at 16 kHz).
/// Only decodes up to `end`, avoiding loading the entire file.
fn decode_time_range(path: &str, start: f32, end: f32) -> Result<Vec<f32>, String> {
    let (mut format_reader, mut decoder, track_id, num_channels, native_rate) =
        open_audio_file(path)?;

    let resampler = if native_rate != 16000 {
        Some(
            LinearResampler::create(native_rate as i32, 16000)
                .ok_or_else(|| format!("Failed to create resampler for {native_rate} Hz"))?,
        )
    } else {
        None
    };

    let rate_16k = 16000.0;
    let start_sample = (start * rate_16k) as usize;
    let end_sample = (end * rate_16k) as usize;
    let mut result: Vec<f32> = Vec::new();
    let mut total_16k_samples: usize = 0;

    loop {
        if total_16k_samples >= end_sample {
            break;
        }

        let packet = match format_reader.next_packet() {
            Ok(p) => p,
            Err(symphonia::core::errors::Error::IoError(e))
                if e.kind() == std::io::ErrorKind::UnexpectedEof =>
            {
                break;
            }
            Err(symphonia::core::errors::Error::ResetRequired) => break,
            Err(err) => return Err(format!("Read error: {err}")),
        };

        if packet.track_id() != track_id {
            continue;
        }

        let decoded = match decoder.decode(&packet) {
            Ok(d) => d,
            Err(symphonia::core::errors::Error::IoError(_))
            | Err(symphonia::core::errors::Error::DecodeError(_)) => continue,
            Err(err) => return Err(format!("Decode error: {err}")),
        };

        let mono = decode_to_mono_f32(&decoded, num_channels);

        let pcm = if let Some(ref resamp) = resampler {
            resamp.resample(&mono, false)
        } else {
            mono
        };

        let chunk_start = total_16k_samples;
        let chunk_end = total_16k_samples + pcm.len();
        total_16k_samples = chunk_end;

        // Only keep samples within [start_sample, end_sample)
        if chunk_end <= start_sample {
            continue;
        }

        let copy_start = if chunk_start >= start_sample {
            0
        } else {
            start_sample - chunk_start
        };
        let copy_end = (end_sample - chunk_start).min(pcm.len());

        if copy_start < copy_end {
            result.extend_from_slice(&pcm[copy_start..copy_end]);
        }
    }

    if result.is_empty() {
        return Err("No audio data in the specified time range".to_string());
    }

    Ok(result)
}

/// Save a single audio segment as a WAV file.
/// Re-decodes only the needed portion from the file (no full-file buffering).
#[tauri::command]
fn save_segment_as_wav(
    path: String,
    start: f32,
    end: f32,
    state: tauri::State<'_, AppState>,
) -> Result<(), String> {
    let audio_path = state.audio_path.lock().map_err(|e| e.to_string())?;
    if audio_path.is_empty() {
        return Err("No audio file has been processed".to_string());
    }

    let samples = decode_time_range(&audio_path, start, end)?;
    write_wav(&path, &samples)
}

/// Return init status as (status_code, error_message).
/// status_code: 0 = pending, 1 = ready, 2 = error.
#[derive(Serialize, Clone)]
struct InitStatus {
    status: u8,
    error: String,
    num_threads: u32,
}

#[tauri::command]
fn get_init_status(state: tauri::State<'_, AppState>) -> InitStatus {
    let status = state.init_status.load(Ordering::Relaxed);
    let error = state.init_error.lock().unwrap().clone();
    let num_threads = state.num_threads.load(Ordering::Relaxed);
    InitStatus {
        status,
        error,
        num_threads,
    }
}

/// Resolve the resource directory where bundled models live.
/// macOS .app bundle: <App>.app/Contents/Resources/
/// Linux / Windows: directory alongside the executable.
fn resource_dir() -> PathBuf {
    #[cfg(target_os = "macos")]
    {
        if let Ok(exe) = std::env::current_exe() {
            if let Some(app_dir) = exe
                .parent()
                .and_then(|p| p.parent())
                .and_then(|p| p.parent())
            {
                let resources = app_dir.join("Resources");
                if resources.exists() {
                    return resources;
                }
            }
        }
    }

    std::env::current_exe()
        .ok()
        .and_then(|p| p.parent().map(|p| p.to_path_buf()))
        .unwrap_or_else(|| PathBuf::from("."))
}

/// Initialize recognizer and VAD. Returns (recognizer, vad, num_threads) or error string.
fn build_models() -> Result<(OfflineRecognizer, VoiceActivityDetector, u32), String> {
    let dir = resource_dir();
    let model_dir = dir.join(MODEL_NAME);

    let mut asr_config = get_model_config(MODEL_TYPE, &model_dir)
        .ok_or_else(|| format!("Unknown MODEL_TYPE: {MODEL_TYPE}"))?;

    // Optional homophone replacer files live in resource_dir(), not model_dir.
    let hr_lexicon = dir.join("lexicon.txt");
    if hr_lexicon.exists() {
        asr_config.hr.lexicon = hr_lexicon.to_str().map(|s| s.to_string());
    }
    let hr_rule_fst = dir.join("replace.fst");
    if hr_rule_fst.exists() {
        asr_config.hr.rule_fsts = hr_rule_fst.to_str().map(|s| s.to_string());
    }

    let num_threads = asr_config.model_config.num_threads as u32;

    let silero_vad_path = dir
        .join("silero_vad.onnx")
        .to_str()
        .ok_or_else(|| "Invalid silero_vad path".to_string())?
        .to_string();

    let mut vad_config = VadModelConfig::default();
    vad_config.silero_vad = SileroVadModelConfig {
        model: Some(silero_vad_path),
        threshold: 0.2,
        min_silence_duration: 0.2,
        min_speech_duration: 0.2,
        window_size: 512,
        max_speech_duration: 10.0,
        ..Default::default()
    };
    vad_config.sample_rate = 16000;
    vad_config.num_threads = 1;

    let recognizer = OfflineRecognizer::create(&asr_config)
        .ok_or_else(|| "Failed to create recognizer. Check model paths.".to_string())?;

    let vad = VoiceActivityDetector::create(&vad_config, 120.0)
        .ok_or_else(|| "Failed to create VAD. Check silero_vad model path.".to_string())?;

    Ok((recognizer, vad, num_threads))
}

/// Create AppState with recognizer/VAD set to None.
/// Models are loaded in a background thread after startup.
fn build_app_state() -> AppState {
    AppState {
        recognizer: Arc::new(Mutex::new(None)),
        vad: Arc::new(Mutex::new(None)),
        cancelled: Arc::new(AtomicBool::new(false)),
        progress: Arc::new(AtomicU32::new(0)),
        status: Arc::new(Mutex::new("idle".to_string())),
        segments: Arc::new(Mutex::new(Vec::new())),
        audio_path: Arc::new(Mutex::new(String::new())),
        init_status: Arc::new(AtomicU8::new(0)), // 0 = pending
        init_error: Arc::new(Mutex::new(String::new())),
        elapsed_ms: Arc::new(AtomicU32::new(0)),
        audio_duration_ms: Arc::new(AtomicU32::new(0)),
        num_threads: Arc::new(AtomicU32::new(0)),
    }
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    let state = build_app_state();

    // Clone Arc handles for the init thread
    let init_recognizer = Arc::clone(&state.recognizer);
    let init_vad = Arc::clone(&state.vad);
    let init_status = Arc::clone(&state.init_status);
    let init_error = Arc::clone(&state.init_error);
    let init_num_threads = Arc::clone(&state.num_threads);

    std::thread::spawn(move || {
        match build_models() {
            Ok((rec, vad, threads)) => {
                *init_recognizer.lock().unwrap() = Some(rec);
                *init_vad.lock().unwrap() = Some(vad);
                init_num_threads.store(threads, Ordering::Relaxed);
                init_status.store(1, Ordering::Relaxed); // ready
            }
            Err(e) => {
                *init_error.lock().unwrap() = e;
                init_status.store(2, Ordering::Relaxed); // error
            }
        }
    });

    tauri::Builder::default()
        .plugin(tauri_plugin_dialog::init())
        .manage(state)
        .invoke_handler(tauri::generate_handler![
            recognize_file,
            get_recognition_progress,
            cancel_recognition,
            export_srt,
            save_segment_as_wav,
            get_init_status,
        ])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
