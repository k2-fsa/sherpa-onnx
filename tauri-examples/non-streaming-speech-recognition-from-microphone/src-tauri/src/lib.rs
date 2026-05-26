mod model_registry;

use model_registry::get_model_config;
use serde::{Deserialize, Serialize};
use sherpa_onnx::{
    LinearResampler, OfflineRecognizer, SileroVadModelConfig, VadModelConfig,
    VoiceActivityDetector,
};
use std::fs::File;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU8, Ordering};
use std::sync::mpsc;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use chrono::Local;
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use cpal::SampleFormat;

/// Which ASR model to bundle. Build scripts patch these via sed.
const MODEL_TYPE: u32 = 15;
const MODEL_NAME: &str = "sherpa-onnx-sense-voice-zh-en-ja-ko-yue-int8-2024-07-17";

#[derive(Serialize, Deserialize, Clone, PartialEq)]
struct VadSettings {
    threshold: f32,
    min_silence_duration: f32,
    min_speech_duration: f32,
    max_speech_duration: f32,
    num_threads: i32,
}

impl Default for VadSettings {
    fn default() -> Self {
        Self {
            threshold: 0.2,
            min_silence_duration: 0.2,
            min_speech_duration: 0.2,
            max_speech_duration: 10.0,
            num_threads: 2,
        }
    }
}

#[derive(Serialize, Clone)]
struct SegmentResult {
    start: f32,
    end: f32,
    wall_start: String,
    wall_end: String,
    text: String,
}

#[derive(Serialize, Clone)]
struct RecordingState {
    recording: bool,
    segments: Vec<SegmentResult>,
    elapsed_secs: f32,
}

struct AppState {
    recognizer: Arc<Mutex<Option<OfflineRecognizer>>>,
    vad: Arc<Mutex<Option<VoiceActivityDetector>>>,
    recording: Arc<AtomicBool>,
    stop_signal: Arc<AtomicBool>,
    segments: Arc<Mutex<Vec<SegmentResult>>>,
    recorded_audio: Arc<Mutex<Vec<f32>>>,
    start_wall_clock: Arc<Mutex<Option<chrono::DateTime<Local>>>>,
    start_instant: Arc<Mutex<Option<Instant>>>,
    init_status: Arc<AtomicU8>,
    init_error: Arc<Mutex<String>>,
    num_threads: Arc<AtomicU32>,
    settings: Arc<Mutex<VadSettings>>,
    selected_device: Arc<Mutex<Option<String>>>,
}

// ---------------------------------------------------------------------------
// cpal microphone capture
// ---------------------------------------------------------------------------

fn build_input_stream(
    device: &cpal::Device,
    tx: mpsc::Sender<Vec<f32>>,
) -> Result<cpal::Stream, String> {
    let supported = device
        .default_input_config()
        .map_err(|e| format!("No input config: {e}"))?;
    let config = supported.config();
    let sample_format = supported.sample_format();
    let channels = config.channels as usize;
    if channels == 0 {
        return Err("Device reports 0 channels".to_string());
    }

    eprintln!(
        "[mic] format: {:?}, channels: {}, sample_rate: {}",
        sample_format, channels, config.sample_rate.0
    );

    let err_fn = |err| eprintln!("[mic] stream error: {:?}", err);

    let stream = match sample_format {
        SampleFormat::F32 => device
            .build_input_stream(
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
            )
            .map_err(|e| format!("Build F32 stream: {e}"))?,

        SampleFormat::I16 => device
            .build_input_stream(
                &config,
                move |data: &[i16], _| {
                    if data.is_empty() {
                        return;
                    }
                    let mono: Vec<f32> = data
                        .chunks(channels)
                        .map(|frame| {
                            let sum: f32 =
                                frame.iter().map(|&s| s as f32 / i16::MAX as f32).sum();
                            sum / channels as f32
                        })
                        .collect();
                    let _ = tx.send(mono);
                },
                err_fn,
                None,
            )
            .map_err(|e| format!("Build I16 stream: {e}"))?,

        SampleFormat::U16 => device
            .build_input_stream(
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
                                .map(|&s| (s as f32 - 32768.0) / 32768.0)
                                .sum();
                            sum / channels as f32
                        })
                        .collect();
                    let _ = tx.send(mono);
                },
                err_fn,
                None,
            )
            .map_err(|e| format!("Build U16 stream: {e}"))?,

        other => return Err(format!("Unsupported sample format: {:?}", other)),
    };

    Ok(stream)
}

// ---------------------------------------------------------------------------
// Recognize a single VAD speech segment
// ---------------------------------------------------------------------------

fn recognize_segment(
    recognizer: &OfflineRecognizer,
    segment: &sherpa_onnx::SpeechSegment,
    segments: &Arc<Mutex<Vec<SegmentResult>>>,
    base_wall: &chrono::DateTime<Local>,
    audio_offset_samples: usize,
) {
    let samples = segment.samples();
    let duration = samples.len() as f32 / 16000.0;
    if duration < 0.1 {
        return;
    }

    let vad_start = segment.start() as f32 / 16000.0;
    let offset_secs = audio_offset_samples as f32 / 16000.0;
    let rel_start = offset_secs + vad_start;
    let rel_end = rel_start + duration;

    let stream = recognizer.create_stream();
    stream.accept_waveform(16000, samples);
    recognizer.decode(&stream);

    if let Some(r) = stream.get_result() {
        let text = r.text.trim().to_string();
        if !text.is_empty()
            && !text
                .chars()
                .all(|c| c.is_ascii_punctuation() || c.is_ascii_whitespace())
        {
            let wall_start = *base_wall
                + chrono::Duration::milliseconds((vad_start * 1000.0) as i64);
            let wall_end =
                *base_wall + chrono::Duration::milliseconds(((vad_start + duration) * 1000.0) as i64);

            if let Ok(mut segs) = segments.lock() {
                segs.push(SegmentResult {
                    start: rel_start,
                    end: rel_end,
                    wall_start: wall_start.format("%Y-%m-%d %H:%M:%S").to_string(),
                    wall_end: wall_end.format("%Y-%m-%d %H:%M:%S").to_string(),
                    text,
                });
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Device enumeration
// ---------------------------------------------------------------------------

#[derive(Serialize, Clone)]
struct InputDevice {
    name: String,
    is_default: bool,
}

#[tauri::command]
fn list_input_devices() -> Result<Vec<InputDevice>, String> {
    let host = cpal::default_host();
    let default_name = host
        .default_input_device()
        .and_then(|d| d.name().ok())
        .unwrap_or_default();

    let devices: Vec<InputDevice> = host
        .input_devices()
        .map_err(|e| format!("Cannot enumerate devices: {e}"))?
        .filter_map(|d| {
            let name = d.name().ok()?;
            Some(InputDevice {
                is_default: name == default_name,
                name,
            })
        })
        .collect();

    Ok(devices)
}

#[tauri::command]
fn set_input_device(
    device_name: Option<String>,
    state: tauri::State<'_, AppState>,
) -> Result<(), String> {
    if state.recording.load(Ordering::SeqCst) {
        return Err("Cannot change device while recording".to_string());
    }
    *state.selected_device.lock().map_err(|e| e.to_string())? = device_name;
    Ok(())
}

#[tauri::command]
fn get_selected_device(state: tauri::State<'_, AppState>) -> Result<Option<String>, String> {
    state
        .selected_device
        .lock()
        .map(|d| d.clone())
        .map_err(|e| e.to_string())
}

// ---------------------------------------------------------------------------
// Tauri commands
// ---------------------------------------------------------------------------

#[tauri::command]
fn start_recording(state: tauri::State<'_, AppState>) -> Result<(), String> {
    if state.recording.swap(true, Ordering::SeqCst) {
        return Err("Already recording".to_string());
    }

    let init = state.init_status.load(Ordering::Relaxed);
    if init != 1 {
        state.recording.store(false, Ordering::SeqCst);
        return Err("Models not ready".to_string());
    }

    // Take recognizer and VAD out of shared state for exclusive use
    let recognizer = {
        let mut guard = state.recognizer.lock().map_err(|e| e.to_string())?;
        guard.take().ok_or("Recognizer not available")?
    };
    let vad = {
        let mut guard = state.vad.lock().map_err(|e| e.to_string())?;
        guard.take().ok_or("VAD not available")?
    };

    state.stop_signal.store(false, Ordering::Relaxed);

    // Get current audio length as offset for new segments
    let audio_offset = state
        .recorded_audio
        .lock()
        .map_err(|e| e.to_string())?
        .len();

    let now = Local::now();
    *state
        .start_wall_clock
        .lock()
        .map_err(|e| e.to_string())? = Some(now);
    *state.start_instant.lock().map_err(|e| e.to_string())? = Some(Instant::now());

    eprintln!("[start_recording] starting at {now}");

    let stop_signal = Arc::clone(&state.stop_signal);
    let recording = Arc::clone(&state.recording);
    let segments = Arc::clone(&state.segments);
    let recorded_audio = Arc::clone(&state.recorded_audio);
    let recognizer_arc = Arc::clone(&state.recognizer);
    let vad_arc = Arc::clone(&state.vad);
    let base_wall = now;
    let selected_device = state
        .selected_device
        .lock()
        .map_err(|e| e.to_string())?
        .clone();

    std::thread::spawn(move || {
        let result = run_recording(
            recognizer,
            vad,
            &stop_signal,
            &segments,
            &recorded_audio,
            &base_wall,
            audio_offset,
            selected_device.as_deref(),
        );

        match result {
            Ok((rec, v)) => {
                if let Ok(mut r) = recognizer_arc.lock() {
                    *r = Some(rec);
                }
                if let Ok(mut va) = vad_arc.lock() {
                    *va = Some(v);
                }
            }
            Err(e) => {
                eprintln!("[recording thread] error: {e}");
            }
        }

        recording.store(false, Ordering::SeqCst);
        eprintln!("[recording thread] stopped");
    });

    Ok(())
}

fn run_recording(
    recognizer: OfflineRecognizer,
    vad: VoiceActivityDetector,
    stop_signal: &AtomicBool,
    segments: &Arc<Mutex<Vec<SegmentResult>>>,
    recorded_audio: &Arc<Mutex<Vec<f32>>>,
    base_wall: &chrono::DateTime<Local>,
    audio_offset: usize,
    selected_device: Option<&str>,
) -> Result<(OfflineRecognizer, VoiceActivityDetector), String> {
    let host = cpal::default_host();
    let device = if let Some(name) = selected_device {
        host.input_devices()
            .map_err(|e| format!("Cannot enumerate devices: {e}"))?
            .find(|d| d.name().ok().as_deref() == Some(name))
            .ok_or_else(|| format!("Device not found: {name}"))?
    } else {
        host.default_input_device()
            .ok_or("No default input device")?
    };

    eprintln!("[recording] device: {:?}", device.name().unwrap_or_default());

    let supported = device
        .default_input_config()
        .map_err(|e| format!("No input config: {e}"))?;
    let mic_sample_rate = supported.sample_rate().0 as i32;

    let resampler = if mic_sample_rate != 16000 {
        Some(
            LinearResampler::create(mic_sample_rate, 16000)
                .ok_or_else(|| format!("Failed to create resampler for {mic_sample_rate} Hz"))?,
        )
    } else {
        None
    };

    let (tx, rx) = mpsc::channel::<Vec<f32>>();
    let stream = build_input_stream(&device, tx)?;
    stream.play().map_err(|e| format!("Stream play: {e}"))?;

    vad.reset();
    let window_size: usize = 512;
    let mut vad_buf: Vec<f32> = Vec::new();

    loop {
        if stop_signal.load(Ordering::Relaxed) {
            break;
        }

        match rx.recv_timeout(Duration::from_millis(100)) {
            Ok(samples) => {
                let pcm = if let Some(ref resamp) = resampler {
                    resamp.resample(&samples, false)
                } else {
                    samples
                };

                if let Ok(mut audio) = recorded_audio.lock() {
                    audio.extend_from_slice(&pcm);
                }

                vad_buf.extend_from_slice(&pcm);

                while vad_buf.len() >= window_size {
                    vad.accept_waveform(&vad_buf[..window_size]);
                    vad_buf.drain(..window_size);

                    while let Some(segment) = vad.front() {
                        recognize_segment(&recognizer, &segment, segments, base_wall, audio_offset);
                        vad.pop();
                    }
                }
            }
            Err(mpsc::RecvTimeoutError::Timeout) => continue,
            Err(mpsc::RecvTimeoutError::Disconnected) => {
                eprintln!("[recording] audio channel disconnected");
                break;
            }
        }
    }

    // Drop the stream to stop capture before flushing
    drop(stream);

    // Feed any remaining samples (zero-pad to window size)
    if !vad_buf.is_empty() {
        vad_buf.resize(window_size, 0.0);
        vad.accept_waveform(&vad_buf[..window_size]);
    }

    // Flush VAD unconditionally — it may have buffered speech internally
    vad.flush();
    while let Some(segment) = vad.front() {
        recognize_segment(&recognizer, &segment, segments, base_wall, audio_offset);
        vad.pop();
    }

    let seg_count = segments.lock().map(|s| s.len()).unwrap_or(0);
    eprintln!("[recording] flushed, total segments: {seg_count}");

    Ok((recognizer, vad))
}

#[tauri::command]
fn stop_recording(state: tauri::State<'_, AppState>) {
    eprintln!("[stop_recording] signalling stop");
    state.stop_signal.store(true, Ordering::Relaxed);
}

#[tauri::command]
fn clear_results(state: tauri::State<'_, AppState>) -> Result<(), String> {
    if state.recording.load(Ordering::SeqCst) {
        return Err("Cannot clear while recording".to_string());
    }
    state.segments.lock().map_err(|e| e.to_string())?.clear();
    state
        .recorded_audio
        .lock()
        .map_err(|e| e.to_string())?
        .clear();
    *state.start_wall_clock.lock().map_err(|e| e.to_string())? = None;
    *state.start_instant.lock().map_err(|e| e.to_string())? = None;
    eprintln!("[clear_results] cleared all segments and audio");
    Ok(())
}

#[tauri::command]
fn get_recording_state(state: tauri::State<'_, AppState>) -> Result<RecordingState, String> {
    let recording = state.recording.load(Ordering::Relaxed);
    let segments = state.segments.lock().map_err(|e| e.to_string())?.clone();
    let elapsed_secs = state
        .start_instant
        .lock()
        .map_err(|e| e.to_string())?
        .map(|i| i.elapsed().as_secs_f32())
        .unwrap_or(0.0);
    Ok(RecordingState {
        recording,
        segments,
        elapsed_secs,
    })
}

#[tauri::command]
fn save_segment_as_wav(
    path: String,
    start: f32,
    end: f32,
    state: tauri::State<'_, AppState>,
) -> Result<(), String> {
    let audio = state.recorded_audio.lock().map_err(|e| e.to_string())?;
    if audio.is_empty() {
        return Err("No recorded audio".to_string());
    }

    let start_sample = (start * 16000.0) as usize;
    let end_sample = ((end * 16000.0) as usize).min(audio.len());
    if start_sample >= end_sample {
        return Err("Invalid time range".to_string());
    }

    write_wav(&path, &audio[start_sample..end_sample])
}

#[tauri::command]
fn save_all_audio(path: String, state: tauri::State<'_, AppState>) -> Result<(), String> {
    let audio = state.recorded_audio.lock().map_err(|e| e.to_string())?;
    if audio.is_empty() {
        return Err("No recorded audio".to_string());
    }
    write_wav(&path, &audio)
}

#[tauri::command]
fn get_recorded_audio_path(state: tauri::State<'_, AppState>) -> Result<String, String> {
    let audio = state.recorded_audio.lock().map_err(|e| e.to_string())?;
    if audio.is_empty() {
        return Err("No recorded audio".to_string());
    }

    let tmp = std::env::temp_dir().join(format!("sherpa-onnx-mic-{}.wav", std::process::id()));
    let tmp_str = tmp
        .to_str()
        .ok_or("Invalid temp path")?
        .to_string();
    write_wav(&tmp_str, &audio)?;
    eprintln!("[get_recorded_audio_path] wrote {tmp_str} ({} samples)", audio.len());
    Ok(tmp_str)
}

fn format_srt_time(seconds: f32) -> String {
    let total_ms = (seconds * 1000.0) as u64;
    let h = total_ms / 3_600_000;
    let m = (total_ms % 3_600_000) / 60_000;
    let s = (total_ms % 60_000) / 1000;
    let ms = total_ms % 1000;
    format!("{h:02}:{m:02}:{s:02},{ms:03}")
}

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

    let f = File::create(path).map_err(|e| format!("Cannot create file: {e}"))?;
    let mut w = std::io::BufWriter::new(f);

    use std::io::Write;
    w.write_all(b"RIFF").map_err(|e| e.to_string())?;
    w.write_all(&file_size.to_le_bytes())
        .map_err(|e| e.to_string())?;
    w.write_all(b"WAVE").map_err(|e| e.to_string())?;
    w.write_all(b"fmt ").map_err(|e| e.to_string())?;
    w.write_all(&16u32.to_le_bytes())
        .map_err(|e| e.to_string())?;
    w.write_all(&1u16.to_le_bytes())
        .map_err(|e| e.to_string())?;
    w.write_all(&1u16.to_le_bytes())
        .map_err(|e| e.to_string())?;
    w.write_all(&16000u32.to_le_bytes())
        .map_err(|e| e.to_string())?;
    w.write_all(&byte_rate.to_le_bytes())
        .map_err(|e| e.to_string())?;
    w.write_all(&2u16.to_le_bytes())
        .map_err(|e| e.to_string())?;
    w.write_all(&16u16.to_le_bytes())
        .map_err(|e| e.to_string())?;
    w.write_all(b"data").map_err(|e| e.to_string())?;
    w.write_all(&data_size.to_le_bytes())
        .map_err(|e| e.to_string())?;
    for &s in samples {
        let clamped = s.max(-1.0).min(1.0);
        let pcm = (clamped * 32767.0) as i16;
        w.write_all(&pcm.to_le_bytes())
            .map_err(|e| e.to_string())?;
    }
    w.flush().map_err(|e| e.to_string())?;

    Ok(())
}

// ---------------------------------------------------------------------------
// Init status
// ---------------------------------------------------------------------------

#[derive(Serialize, Clone)]
struct InitStatus {
    status: u8,
    error: String,
    num_threads: u32,
}

#[tauri::command]
fn get_init_status(state: tauri::State<'_, AppState>) -> InitStatus {
    let status = state.init_status.load(Ordering::Relaxed);
    let error = state
        .init_error
        .lock()
        .map(|e| e.clone())
        .unwrap_or_default();
    let num_threads = state.num_threads.load(Ordering::Relaxed);
    InitStatus {
        status,
        error,
        num_threads,
    }
}

// ---------------------------------------------------------------------------
// Settings
// ---------------------------------------------------------------------------

#[tauri::command]
fn get_settings(state: tauri::State<'_, AppState>) -> Result<VadSettings, String> {
    state
        .settings
        .lock()
        .map(|s| s.clone())
        .map_err(|e| e.to_string())
}

fn validate_settings(s: &VadSettings) -> Result<(), String> {
    if s.threshold <= 0.0 || s.threshold >= 1.0 {
        return Err("threshold must be between 0.0 and 1.0 (exclusive)".to_string());
    }
    if s.min_silence_duration < 0.0 {
        return Err("min_silence_duration must be >= 0".to_string());
    }
    if s.min_speech_duration < 0.0 {
        return Err("min_speech_duration must be >= 0".to_string());
    }
    if s.max_speech_duration <= 0.0 {
        return Err("max_speech_duration must be > 0".to_string());
    }
    if s.num_threads < 1 || s.num_threads > 16 {
        return Err("num_threads must be between 1 and 16".to_string());
    }
    Ok(())
}

#[tauri::command]
fn apply_settings(
    new_settings: VadSettings,
    state: tauri::State<'_, AppState>,
) -> Result<(), String> {
    if state.recording.load(Ordering::SeqCst) {
        return Err("Cannot change settings while recording".to_string());
    }
    let init = state.init_status.load(Ordering::Relaxed);
    if init == 0 {
        return Err("Models are still loading, please wait".to_string());
    }

    validate_settings(&new_settings)?;

    {
        let current = state.settings.lock().map_err(|e| e.to_string())?;
        if *current == new_settings {
            return Ok(());
        }
    }

    state.init_status.store(0, Ordering::Relaxed);
    *state.settings.lock().map_err(|e| e.to_string())? = new_settings.clone();

    let recognizer_arc = Arc::clone(&state.recognizer);
    let vad_arc = Arc::clone(&state.vad);
    let init_status = Arc::clone(&state.init_status);
    let init_error = Arc::clone(&state.init_error);
    let init_num_threads = Arc::clone(&state.num_threads);

    std::thread::spawn(move || {
        eprintln!("[apply_settings] rebuilding models...");
        match build_models(&new_settings) {
            Ok((rec, vad, threads)) => {
                eprintln!("[apply_settings] models rebuilt, num_threads={threads}");
                let r_ok = recognizer_arc.lock().map(|mut r| { *r = Some(rec); }).is_ok();
                let v_ok = vad_arc.lock().map(|mut v| { *v = Some(vad); }).is_ok();
                if r_ok && v_ok {
                    init_num_threads.store(threads, Ordering::Relaxed);
                    init_status.store(1, Ordering::Relaxed);
                } else {
                    eprintln!("[apply_settings] mutex poisoned, marking as error");
                    if let Ok(mut err) = init_error.lock() {
                        *err = "Internal error: mutex poisoned".to_string();
                    }
                    init_status.store(2, Ordering::Relaxed);
                }
            }
            Err(e) => {
                eprintln!("[apply_settings] rebuild failed: {e}");
                if let Ok(mut err) = init_error.lock() {
                    *err = e;
                }
                init_status.store(2, Ordering::Relaxed);
            }
        }
    });

    Ok(())
}

// ---------------------------------------------------------------------------
// Resource directory & model init
// ---------------------------------------------------------------------------

fn resource_dir() -> PathBuf {
    if let Ok(exe) = std::env::current_exe() {
        eprintln!("[resource_dir] current_exe: {exe:?}");

        for ancestor in exe.ancestors() {
            if ancestor
                .extension()
                .map_or(false, |ext| ext == "app")
            {
                let resources = ancestor.join("Contents").join("Resources");
                eprintln!("[resource_dir] found .app bundle: {ancestor:?}");
                if resources.exists() {
                    let assets = resources.join("assets");
                    if assets.exists() {
                        eprintln!("[resource_dir] using assets inside Resources: {assets:?}");
                        return assets;
                    }
                    eprintln!("[resource_dir] using Resources directly: {resources:?}");
                    return resources;
                }
                break;
            }
        }

        if let Some(exe_dir) = exe.parent() {
            let assets_dir = exe_dir.join("assets");
            if assets_dir.exists() {
                eprintln!("[resource_dir] using assets dir: {assets_dir:?}");
                return assets_dir;
            }
            eprintln!("[resource_dir] using exe dir: {exe_dir:?}");
            return exe_dir.to_path_buf();
        }
    }
    eprintln!("[resource_dir] fallback to current directory");
    PathBuf::from(".")
}

fn build_models(
    settings: &VadSettings,
) -> Result<(OfflineRecognizer, VoiceActivityDetector, u32), String> {
    let dir = resource_dir();
    let model_dir = dir.join(MODEL_NAME);
    let silero_vad_path = dir.join("silero_vad.onnx");

    eprintln!("[build_models] MODEL_TYPE={MODEL_TYPE}, MODEL_NAME={MODEL_NAME}");
    eprintln!("[build_models] resource_dir: {dir:?}");
    eprintln!(
        "[build_models] model_dir: {model_dir:?}, exists={}",
        model_dir.exists()
    );
    if model_dir.exists() {
        if let Ok(entries) = std::fs::read_dir(&model_dir) {
            for entry in entries.flatten() {
                eprintln!("[build_models]   model_dir entry: {:?}", entry.path());
            }
        }
    } else {
        eprintln!("[build_models] ERROR: model_dir does not exist!");
        eprintln!("[build_models] dir contents:");
        if let Ok(entries) = std::fs::read_dir(&dir) {
            for entry in entries.flatten() {
                eprintln!("[build_models]   {:?}", entry.path());
            }
        } else {
            eprintln!("[build_models]   (cannot read dir)");
        }
    }
    eprintln!(
        "[build_models] silero_vad: {silero_vad_path:?}, exists={}",
        silero_vad_path.exists()
    );

    let mut asr_config = get_model_config(MODEL_TYPE, &model_dir).ok_or_else(|| {
        format!(
            "Unknown MODEL_TYPE: {MODEL_TYPE}. model_dir={model_dir:?}, exists={}",
            model_dir.exists()
        )
    })?;

    eprintln!(
        "[build_models] got ASR config, num_threads={}",
        asr_config.model_config.num_threads
    );

    let hr_lexicon = dir.join("lexicon.txt");
    if hr_lexicon.exists() {
        eprintln!("[build_models] using homophone replacer lexicon: {hr_lexicon:?}");
        asr_config.hr.lexicon = hr_lexicon.to_str().map(|s| s.to_string());
    }
    let hr_rule_fst = dir.join("replace.fst");
    if hr_rule_fst.exists() {
        eprintln!("[build_models] using homophone replacer rule_fst: {hr_rule_fst:?}");
        asr_config.hr.rule_fsts = hr_rule_fst.to_str().map(|s| s.to_string());
    }

    asr_config.model_config.num_threads = settings.num_threads;
    let num_threads = settings.num_threads as u32;

    let silero_vad_str = silero_vad_path
        .to_str()
        .ok_or_else(|| format!("Invalid silero_vad path: {silero_vad_path:?}"))?
        .to_string();

    let mut vad_config = VadModelConfig::default();
    vad_config.silero_vad = SileroVadModelConfig {
        model: Some(silero_vad_str),
        threshold: settings.threshold,
        min_silence_duration: settings.min_silence_duration,
        min_speech_duration: settings.min_speech_duration,
        window_size: 512,
        max_speech_duration: settings.max_speech_duration,
        ..Default::default()
    };
    vad_config.sample_rate = 16000;
    vad_config.num_threads = 1;

    eprintln!("[build_models] creating recognizer...");
    let recognizer = OfflineRecognizer::create(&asr_config).ok_or_else(|| {
        format!(
            "Failed to create recognizer. MODEL_TYPE={MODEL_TYPE}, model_dir={model_dir:?}, \
             dir contents: {:?}",
            std::fs::read_dir(&dir)
                .map(|entries| entries
                    .flatten()
                    .map(|e| e.file_name().to_string_lossy().into_owned())
                    .collect::<Vec<_>>())
                .unwrap_or_default()
        )
    })?;
    eprintln!("[build_models] recognizer created");

    eprintln!("[build_models] creating VAD...");
    let vad = VoiceActivityDetector::create(&vad_config, 120.0).ok_or_else(|| {
        format!(
            "Failed to create VAD. silero_vad={silero_vad_path:?}, exists={}",
            silero_vad_path.exists()
        )
    })?;
    eprintln!("[build_models] VAD created");

    Ok((recognizer, vad, num_threads))
}

// ---------------------------------------------------------------------------
// App startup
// ---------------------------------------------------------------------------

fn build_app_state() -> AppState {
    AppState {
        recognizer: Arc::new(Mutex::new(None)),
        vad: Arc::new(Mutex::new(None)),
        recording: Arc::new(AtomicBool::new(false)),
        stop_signal: Arc::new(AtomicBool::new(false)),
        segments: Arc::new(Mutex::new(Vec::new())),
        recorded_audio: Arc::new(Mutex::new(Vec::new())),
        start_wall_clock: Arc::new(Mutex::new(None)),
        start_instant: Arc::new(Mutex::new(None)),
        init_status: Arc::new(AtomicU8::new(0)),
        init_error: Arc::new(Mutex::new(String::new())),
        num_threads: Arc::new(AtomicU32::new(0)),
        settings: Arc::new(Mutex::new(VadSettings::default())),
        selected_device: Arc::new(Mutex::new(None)),
    }
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    let state = build_app_state();

    let init_recognizer = Arc::clone(&state.recognizer);
    let init_vad = Arc::clone(&state.vad);
    let init_status = Arc::clone(&state.init_status);
    let init_error = Arc::clone(&state.init_error);
    let init_num_threads = Arc::clone(&state.num_threads);
    let init_settings = Arc::clone(&state.settings);

    std::thread::spawn(move || {
        eprintln!("[init] starting model initialization...");
        let settings = init_settings.lock().map(|s| s.clone()).unwrap_or_default();
        match build_models(&settings) {
            Ok((rec, vad, threads)) => {
                eprintln!("[init] models ready, num_threads={threads}");
                let r_ok = init_recognizer.lock().map(|mut r| { *r = Some(rec); }).is_ok();
                let v_ok = init_vad.lock().map(|mut v| { *v = Some(vad); }).is_ok();
                if r_ok && v_ok {
                    init_num_threads.store(threads, Ordering::Relaxed);
                    if let Ok(mut s) = init_settings.lock() {
                        s.num_threads = threads as i32;
                    }
                    init_status.store(1, Ordering::Relaxed);
                } else {
                    eprintln!("[init] mutex poisoned, marking as error");
                    if let Ok(mut err) = init_error.lock() {
                        *err = "Internal error: mutex poisoned".to_string();
                    }
                    init_status.store(2, Ordering::Relaxed);
                }
            }
            Err(e) => {
                eprintln!("[init] model initialization failed: {e}");
                if let Ok(mut err) = init_error.lock() {
                    *err = e;
                }
                init_status.store(2, Ordering::Relaxed);
            }
        }
    });

    tauri::Builder::default()
        .plugin(tauri_plugin_dialog::init())
        .plugin(tauri_plugin_shell::init())
        .manage(state)
        .invoke_handler(tauri::generate_handler![
            list_input_devices,
            set_input_device,
            get_selected_device,
            start_recording,
            stop_recording,
            clear_results,
            get_recording_state,
            save_segment_as_wav,
            save_all_audio,
            get_recorded_audio_path,
            export_srt,
            get_init_status,
            get_settings,
            apply_settings,
        ])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
