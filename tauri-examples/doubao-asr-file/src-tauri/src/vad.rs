// Silero VAD segmentation via sherpa-onnx. Tuning matches the Python
// reference (generate-subtitles-doubao.py).

use sherpa_onnx::{SileroVadModelConfig, TenVadModelConfig, VadModelConfig, VoiceActivityDetector};

pub struct RawSegment {
    /// start time in seconds
    pub start: f32,
    /// end time in seconds
    pub end: f32,
    /// 16 kHz mono samples of this segment
    pub samples: Vec<f32>,
}

/// Run silero-vad over 16 kHz mono samples and return speech segments.
pub fn run_silero_vad(
    samples_16k: &[f32],
    model_path: &str,
) -> Result<Vec<RawSegment>, String> {
    let config = VadModelConfig {
        silero_vad: SileroVadModelConfig {
            model: Some(model_path.to_string()),
            threshold: 0.1,
            min_silence_duration: 0.3,
            min_speech_duration: 0.1,
            max_speech_duration: 15.0,
            window_size: 512,
        },
        ten_vad: TenVadModelConfig {
            model: None,
            ..Default::default()
        },
        sample_rate: 16000,
        num_threads: 1,
        provider: Some("cpu".to_string()),
        debug: false,
    };

    let window_size = config.silero_vad.window_size as usize;
    let vad = VoiceActivityDetector::create(&config, 100.0)
        .ok_or_else(|| format!("Cannot create VAD with model {model_path}"))?;

    let mut out = Vec::new();
    let mut i = 0;
    while i + window_size <= samples_16k.len() {
        vad.accept_waveform(&samples_16k[i..i + window_size]);
        i += window_size;
        drain(&vad, &mut out);
    }
    vad.flush();
    drain(&vad, &mut out);
    Ok(out)
}

/// Move all completed VAD segments out of the detector.
fn drain(vad: &VoiceActivityDetector, out: &mut Vec<RawSegment>) {
    while !vad.is_empty() {
        if let Some(seg) = vad.front() {
            // seg.start() is a 64-bit sample index (i64 in current sherpa-onnx,
            // i32 in older releases) — convert through f64 so long files keep
            // exact positions before narrowing to f32 seconds.
            let start = seg.start() as f64 / 16000.0;
            let samples = seg.samples().to_vec();
            let end = start + samples.len() as f64 / 16000.0;
            out.push(RawSegment {
                start: start as f32,
                end: end as f32,
                samples,
            });
        }
        vad.pop();
    }
}
