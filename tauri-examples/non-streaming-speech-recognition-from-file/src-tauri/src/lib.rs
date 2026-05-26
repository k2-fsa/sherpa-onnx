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
use std::sync::{Arc, Mutex};
use symphonia::core::audio::{AudioBufferRef, SampleBuffer};
use symphonia::core::codecs::{Decoder, CODEC_TYPE_NULL};
use symphonia::core::formats::FormatReader;
use symphonia::core::io::MediaSourceStream;
use symphonia::core::probe::Hint;

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
    text: String,
}

#[derive(Serialize, Clone, Default)]
struct ProcessingState {
    percent: u32,
    status: String,
    segments: Vec<SegmentResult>,
    elapsed_secs: f32,
    audio_duration_secs: f32,
}

/// Shared application state, created once at startup.
/// Uses Arc so processing thread can hold references.
/// Recognizer and VAD are Option because they are initialized in a background thread.
/// init_status: 0 = pending, 1 = ready, 2 = error.
struct AppState {
    recognizer: Arc<Mutex<Option<OfflineRecognizer>>>,
    vad: Arc<Mutex<Option<VoiceActivityDetector>>>,
    running: Arc<AtomicBool>,
    cancelled: Arc<AtomicBool>,
    progress: Arc<AtomicU32>,
    status: Arc<Mutex<String>>,
    segments: Arc<Mutex<Vec<SegmentResult>>>,
    audio_path: Arc<Mutex<String>>,
    init_status: Arc<AtomicU8>,
    init_error: Arc<Mutex<String>>,
    num_threads: Arc<AtomicU32>,
    settings: Arc<Mutex<VadSettings>>,
    elapsed_secs: Arc<Mutex<f32>>,
    audio_duration_secs: Arc<Mutex<f32>>,
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

/// Recognize a single VAD speech segment: skip if too short or punctuation-only.
fn recognize_segment(
    recognizer: &OfflineRecognizer,
    segment: &sherpa_onnx::SpeechSegment,
    segments: &Arc<Mutex<Vec<SegmentResult>>>,
) {
    let samples = segment.samples();
    let duration = samples.len() as f32 / 16000.0;
    if duration < 0.1 {
        return;
    }

    let start_time = segment.start() as f32 / 16000.0;
    let end_time = start_time + duration;

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
            if let Ok(mut segs) = segments.lock() {
                segs.push(SegmentResult {
                    start: start_time,
                    end: end_time,
                    text,
                });
            }
        }
    }
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
    if state.running.swap(true, Ordering::SeqCst) {
        return Err("Recognition is already running".to_string());
    }

    // Check initialization
    let init = state.init_status.load(Ordering::Relaxed);
    if init == 0 {
        state.running.store(false, Ordering::SeqCst);
        return Err("Models are still loading, please wait".to_string());
    }
    if init == 2 {
        state.running.store(false, Ordering::SeqCst);
        let err = state.init_error.lock().map_err(|e| e.to_string())?.clone();
        return Err(format!("Initialization failed: {err}"));
    }

    eprintln!("[recognize_file] starting recognition for: {path}");

    // Reset state
    state.cancelled.store(false, Ordering::Relaxed);
    state.progress.store(0, Ordering::Relaxed);
    *state.status.lock().map_err(|e| e.to_string())? = "processing".to_string();
    state.segments.lock().map_err(|e| e.to_string())?.clear();
    *state.audio_path.lock().map_err(|e| e.to_string())? = path.clone();
    *state.elapsed_secs.lock().map_err(|e| e.to_string())? = 0.0;
    *state.audio_duration_secs.lock().map_err(|e| e.to_string())? = 0.0;

    // Clone Arc handles for the worker thread
    let recognizer = Arc::clone(&state.recognizer);
    let vad = Arc::clone(&state.vad);
    let running = Arc::clone(&state.running);
    let cancelled = Arc::clone(&state.cancelled);
    let progress = Arc::clone(&state.progress);
    let status = Arc::clone(&state.status);
    let segments = Arc::clone(&state.segments);
    let elapsed_secs = Arc::clone(&state.elapsed_secs);
    let audio_duration_secs = Arc::clone(&state.audio_duration_secs);

    std::thread::spawn(move || {
        let start_time = std::time::Instant::now();
        let result = run_recognition(
            &path,
            &recognizer,
            &vad,
            &cancelled,
            &progress,
            &segments,
        );
        let elapsed = start_time.elapsed().as_secs_f32();

        if let Ok(mut e) = elapsed_secs.lock() {
            *e = elapsed;
        }

        let Ok(mut s) = status.lock() else {
            running.store(false, Ordering::SeqCst);
            return;
        };
        match result {
            Ok(audio_dur) => {
                if let Ok(mut d) = audio_duration_secs.lock() {
                    *d = audio_dur;
                }
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
        running.store(false, Ordering::SeqCst);
    });

    Ok(())
}

/// Stream audio through VAD + ASR without buffering the entire file.
/// Returns the audio duration in seconds on success.
fn run_recognition(
    path: &str,
    recognizer: &Arc<Mutex<Option<OfflineRecognizer>>>,
    vad: &Arc<Mutex<Option<VoiceActivityDetector>>>,
    cancelled: &AtomicBool,
    progress: &Arc<AtomicU32>,
    segments: &Arc<Mutex<Vec<SegmentResult>>>,
) -> Result<f32, String> {
    let (mut format_reader, mut decoder, track_id, num_channels, native_rate) =
        open_audio_file(path)?;

    eprintln!("[run_recognition] file: {path}, native_rate={native_rate}, channels={num_channels}");

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

    let mut recognizer_guard = recognizer.lock().map_err(|e| e.to_string())?;
    let recognizer = recognizer_guard.as_mut().ok_or("Recognizer not initialized")?;
    let mut vad_guard = vad.lock().map_err(|e| e.to_string())?;
    let vad = vad_guard.as_mut().ok_or("VAD not initialized")?;
    vad.reset();

    let window_size: usize = 512;
    let mut vad_buf: Vec<f32> = Vec::new();
    let mut decoded_count: usize = 0;
    let mut last_progress: u32 = 0;

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
                return Ok(decoded_count as f32 / native_rate as f32);
            }

            vad.accept_waveform(&vad_buf[..window_size]);
            vad_buf.drain(..window_size);

            while let Some(segment) = vad.front() {
                recognize_segment(recognizer, &segment, segments);
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
    }

    // Flush remaining samples through VAD
    if !cancelled.load(Ordering::Relaxed) {
        if !vad_buf.is_empty() {
            vad_buf.resize(window_size, 0.0);
            vad.accept_waveform(&vad_buf[..window_size]);
        }
        vad.flush();
        while let Some(segment) = vad.front() {
            recognize_segment(recognizer, &segment, segments);
            vad.pop();
        }
    }

    let audio_duration = decoded_count as f32 / native_rate as f32;
    let final_count = segments.lock().map(|s| s.len()).unwrap_or(0);
    eprintln!("[run_recognition] done, total segments: {final_count}, audio_duration: {audio_duration:.2}s");

    Ok(audio_duration)
}

/// Poll this from the frontend to get current progress and results.
#[tauri::command]
fn get_recognition_progress(state: tauri::State<'_, AppState>) -> Result<ProcessingState, String> {
    let percent = state.progress.load(Ordering::Relaxed);
    let status = state.status.lock().map_err(|e| e.to_string())?.clone();
    let segments = state.segments.lock().map_err(|e| e.to_string())?.clone();
    let elapsed_secs = *state.elapsed_secs.lock().map_err(|e| e.to_string())?;
    let audio_duration_secs = *state.audio_duration_secs.lock().map_err(|e| e.to_string())?;
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

    let f = File::create(path).map_err(|e| format!("Cannot create file: {e}"))?;
    let mut w = std::io::BufWriter::new(f);

    use std::io::Write;
    w.write_all(b"RIFF").map_err(|e| e.to_string())?;
    w.write_all(&file_size.to_le_bytes()).map_err(|e| e.to_string())?;
    w.write_all(b"WAVE").map_err(|e| e.to_string())?;
    w.write_all(b"fmt ").map_err(|e| e.to_string())?;
    w.write_all(&16u32.to_le_bytes()).map_err(|e| e.to_string())?;
    w.write_all(&1u16.to_le_bytes()).map_err(|e| e.to_string())?;
    w.write_all(&1u16.to_le_bytes()).map_err(|e| e.to_string())?;
    w.write_all(&16000u32.to_le_bytes()).map_err(|e| e.to_string())?;
    w.write_all(&byte_rate.to_le_bytes()).map_err(|e| e.to_string())?;
    w.write_all(&2u16.to_le_bytes()).map_err(|e| e.to_string())?;
    w.write_all(&16u16.to_le_bytes()).map_err(|e| e.to_string())?;
    w.write_all(b"data").map_err(|e| e.to_string())?;
    w.write_all(&data_size.to_le_bytes()).map_err(|e| e.to_string())?;
    for &s in samples {
        let clamped = s.max(-1.0).min(1.0);
        let pcm = (clamped * 32767.0) as i16;
        w.write_all(&pcm.to_le_bytes()).map_err(|e| e.to_string())?;
    }
    w.flush().map_err(|e| e.to_string())?;

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

/// Resolve the resource directory where bundled models live.
/// macOS .app bundle: <App>.app/Contents/Resources/
/// Linux / Windows: directory alongside the executable.
fn resource_dir() -> PathBuf {
    if let Ok(exe) = std::env::current_exe() {
        eprintln!("[resource_dir] current_exe: {exe:?}");

        // On macOS, walk up ancestors to find the .app bundle
        for ancestor in exe.ancestors() {
            if ancestor
                .extension()
                .map_or(false, |ext| ext == "app")
            {
                let resources = ancestor.join("Contents").join("Resources");
                eprintln!("[resource_dir] found .app bundle: {ancestor:?}");
                if resources.exists() {
                    // Tauri v2 copies the assets/ dir into Contents/Resources/
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

        // On Windows/Linux, check for assets/ subdirectory next to the exe
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

/// Initialize recognizer and VAD. Returns (recognizer, vad, num_threads) or error string.
fn build_models(settings: &VadSettings) -> Result<(OfflineRecognizer, VoiceActivityDetector, u32), String> {
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

    // Optional homophone replacer files live in resource_dir(), not model_dir.
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
fn get_settings(state: tauri::State<'_, AppState>) -> Result<VadSettings, String> {
    state.settings.lock().map(|s| s.clone()).map_err(|e| e.to_string())
}

#[tauri::command]
fn apply_settings(
    new_settings: VadSettings,
    state: tauri::State<'_, AppState>,
) -> Result<(), String> {
    if state.running.load(Ordering::SeqCst) {
        return Err("Cannot change settings while recognition is running".to_string());
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

    // Set init_status to 0 so the frontend shows "Loading models..."
    state.init_status.store(0, Ordering::Relaxed);

    // Store the new settings
    *state.settings.lock().map_err(|e| e.to_string())? = new_settings.clone();

    let recognizer_arc = Arc::clone(&state.recognizer);
    let vad_arc = Arc::clone(&state.vad);
    let init_status = Arc::clone(&state.init_status);
    let init_error = Arc::clone(&state.init_error);
    let init_num_threads = Arc::clone(&state.num_threads);

    std::thread::spawn(move || {
        eprintln!("[apply_settings] rebuilding models with new settings...");
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

/// Create AppState with recognizer/VAD set to None.
/// Models are loaded in a background thread after startup.
fn build_app_state() -> AppState {
    AppState {
        recognizer: Arc::new(Mutex::new(None)),
        vad: Arc::new(Mutex::new(None)),
        running: Arc::new(AtomicBool::new(false)),
        cancelled: Arc::new(AtomicBool::new(false)),
        progress: Arc::new(AtomicU32::new(0)),
        status: Arc::new(Mutex::new("idle".to_string())),
        segments: Arc::new(Mutex::new(Vec::new())),
        audio_path: Arc::new(Mutex::new(String::new())),
        init_status: Arc::new(AtomicU8::new(0)), // 0 = pending
        init_error: Arc::new(Mutex::new(String::new())),
        num_threads: Arc::new(AtomicU32::new(0)),
        settings: Arc::new(Mutex::new(VadSettings::default())),
        elapsed_secs: Arc::new(Mutex::new(0.0)),
        audio_duration_secs: Arc::new(Mutex::new(0.0)),
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
            recognize_file,
            get_recognition_progress,
            cancel_recognition,
            export_srt,
            save_segment_as_wav,
            get_init_status,
            get_settings,
            apply_settings,
        ])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
