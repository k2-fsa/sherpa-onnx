use serde::Serialize;
use sherpa_onnx::{
    LinearResampler, OfflineRecognizer, OfflineRecognizerConfig, OfflineSenseVoiceModelConfig,
    SileroVadModelConfig, VadModelConfig, VoiceActivityDetector,
};
use std::fs::File;
use std::sync::Mutex;
use symphonia::core::audio::SampleBuffer;
use symphonia::core::codecs::CODEC_TYPE_NULL;
use symphonia::core::errors::Error;
use symphonia::core::io::MediaSourceStream;
use symphonia::core::probe::Hint;

#[derive(Serialize)]
struct SegmentResult {
    start: f32,
    end: f32,
    text: String,
}

/// Shared application state, created once at startup.
/// Mutex ensures exclusive access to VAD/recognizer during a command.
struct AppState {
    recognizer: Mutex<OfflineRecognizer>,
    vad: Mutex<VoiceActivityDetector>,
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

/// Recognize speech from an audio/video file using VAD + offline ASR.
#[tauri::command]
fn recognize_file(
    path: String,
    state: tauri::State<'_, AppState>,
) -> Result<Vec<SegmentResult>, String> {
    // ------------------------------------------------------------------
    // 1. Decode audio file (MP3, FLAC, AAC, WAV, MP4, MKV, etc.)
    // ------------------------------------------------------------------
    let (samples, sample_rate) = decode_audio_file(&path)?;

    // ------------------------------------------------------------------
    // 2. Resample to 16 kHz if needed (VAD requires 16 kHz)
    // ------------------------------------------------------------------
    let resampled;
    let input_samples = if sample_rate != 16000 {
        let resampler = LinearResampler::create(sample_rate, 16000)
            .ok_or_else(|| format!("Failed to create resampler for {sample_rate} Hz"))?;
        resampled = resampler.resample(&samples, true);
        resampled.as_slice()
    } else {
        samples.as_slice()
    };

    // ------------------------------------------------------------------
    // 3. Run VAD + ASR (reuses the shared recognizer and VAD)
    // ------------------------------------------------------------------
    let recognizer = state.recognizer.lock().map_err(|e| e.to_string())?;
    let vad = state.vad.lock().map_err(|e| e.to_string())?;
    vad.reset();

    let window_size: usize = 512;
    let mut results = Vec::new();

    let mut i = 0;
    while i < input_samples.len() {
        let end = (i + window_size).min(input_samples.len());
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
                        results.push(SegmentResult {
                            start: start_time,
                            end: end_time,
                            text: r.text,
                        });
                    }
                }
            }
            vad.pop();
        }

        i += window_size;
    }

    Ok(results)
}

/// Create the recognizer and VAD once at startup.
fn build_app_state() -> AppState {
    // ------------------------------------------------------------------
    // Model configuration — edit these paths to match your setup
    // ------------------------------------------------------------------
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
        recognizer: Mutex::new(recognizer),
        vad: Mutex::new(vad),
    }
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    let state = build_app_state();

    tauri::Builder::default()
        .plugin(tauri_plugin_dialog::init())
        .manage(state)
        .invoke_handler(tauri::generate_handler![recognize_file])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
