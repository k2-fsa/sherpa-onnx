// Decode an audio or video file to 16 kHz mono f32 samples.
//
// Primary path: symphonia (pure Rust, no system deps, Tauri-packable).
// Fallback: ffmpeg on PATH, for codecs symphonia cannot decode (opus, wma)
// and containers it cannot demux (avi). This matches the input format list of
// the reference app:
//   audio: wav mp3 flac ogg oga m4a aac opus weba wma aiff aif
//   video: mp4 webm mov mkv avi m4v ogv 3gp

use sherpa_onnx::LinearResampler;
use std::io::Read;
use std::path::Path;
use std::process::{Command, Stdio};

pub const TARGET_SAMPLE_RATE: i32 = 16000;

/// Decode a time range of the file to 16 kHz mono samples in [-1, 1].
///
/// `start_sec` is the range start; `end_sec <= 0` means "to the end of the
/// file". The returned samples correspond to `[start_sec, end_sec)` only, so
/// callers should add `start_sec` to segment times to get file-absolute times.
/// Tries symphonia first; falls back to ffmpeg when available.
pub fn decode_to_mono_16k(path: &str, start_sec: f32, end_sec: f32) -> Result<Vec<f32>, String> {
    if !Path::new(path).is_file() {
        return Err(format!("{path} does not exist"));
    }
    let start = if start_sec.is_finite() && start_sec > 0.0 {
        start_sec
    } else {
        0.0
    };
    let end = if end_sec.is_finite() && end_sec > 0.0 {
        end_sec
    } else {
        f32::INFINITY
    };
    if end <= start {
        return Err("Invalid range: the end time must be after the start time".to_string());
    }

    match decode_with_symphonia(path, start, end) {
        Ok(samples) => Ok(samples),
        Err(symphonia_err) => {
            if ffmpeg_available() {
                decode_with_ffmpeg(path, start, end).map_err(|e| {
                    format!(
                        "Cannot decode {path}.\nsymphonia: {symphonia_err}\nffmpeg: {e}"
                    )
                })
            } else {
                Err(format!(
                    "Cannot decode {path}: {symphonia_err}. \
                     The file may not be an audio/video file, or installing \
                     ffmpeg may add support for this format."
                ))
            }
        }
    }
}

/// Decode with symphonia and resample to 16 kHz mono. Only samples inside
/// [start_sec, end_sec) are kept; decoding stops once the range is passed.
fn decode_with_symphonia(path: &str, start_sec: f32, end_sec: f32) -> Result<Vec<f32>, String> {
    use symphonia::core::codecs::CODEC_TYPE_NULL;
    use symphonia::core::io::MediaSourceStream;
    use symphonia::core::probe::Hint;

    let src = std::fs::File::open(path).map_err(|e| format!("Cannot open file: {e}"))?;
    let mss = MediaSourceStream::new(Box::new(src), Default::default());

    let mut hint = Hint::new();
    if let Some(ext) = Path::new(path).extension().and_then(|e| e.to_str()) {
        hint.with_extension(ext);
    }

    let probed = symphonia::default::get_probe()
        .format(&hint, mss, &Default::default(), &Default::default())
        .map_err(|e| format!("Unsupported format: {e}"))?;

    let mut format = probed.format;

    let track_id = {
        let track = format
            .tracks()
            .iter()
            .find(|t| t.codec_params.codec != CODEC_TYPE_NULL)
            .ok_or_else(|| "No supported audio track found".to_string())?;
        track.id
    };

    let mut decoder = symphonia::default::get_codecs()
        .make(
            &format
                .tracks()
                .iter()
                .find(|t| t.id == track_id)
                .unwrap()
                .codec_params,
            &Default::default(),
        )
        .map_err(|e| format!("Cannot create decoder: {e}"))?;

    let mut samples: Vec<f32> = Vec::new();
    let mut produced: usize = 0; // mono samples at the native sample rate
    // Sample rate and channel count come from the DECODED buffers: some
    // containers leave codec_params empty until the first packet is decoded,
    // and trusting the fallbacks would distort stereo audio and every
    // timestamp. Range offsets are computed once the real rate is known.
    let mut rate: Option<u32> = None;
    let mut skip = 0usize;
    let mut take = usize::MAX;

    loop {
        if rate.is_some() && produced >= skip.saturating_add(take) {
            break;
        }
        let packet = match format.next_packet() {
            Ok(p) => p,
            Err(symphonia::core::errors::Error::IoError(e))
                if e.kind() == std::io::ErrorKind::UnexpectedEof =>
            {
                break
            }
            Err(symphonia::core::errors::Error::ResetRequired) => break,
            Err(e) => return Err(format!("Decode error: {e}")),
        };
        if packet.track_id() != track_id {
            continue;
        }
        match decoder.decode(&packet) {
            Ok(decoded) => {
                let spec = *decoded.spec();
                let channels = spec.channels.count().max(1);
                if rate.is_none() {
                    let r = spec.rate;
                    rate = Some(r);
                    skip = (start_sec * r as f32) as usize;
                    take = if end_sec.is_finite() {
                        (((end_sec - start_sec) * r as f32) as usize).max(1)
                    } else {
                        usize::MAX
                    };
                }
                let mono = decode_to_mono_f32(&decoded, channels);
                let chunk_start = produced;
                let chunk_end = produced + mono.len();
                produced = chunk_end;

                // Keep only the part of this chunk that overlaps [skip, skip+take).
                let lo = skip.max(chunk_start);
                let hi = (skip.saturating_add(take)).min(chunk_end);
                if lo < hi {
                    samples.extend_from_slice(&mono[lo - chunk_start..hi - chunk_start]);
                }
            }
            Err(symphonia::core::errors::Error::DecodeError(_)) => continue,
            Err(e) => return Err(format!("Decode error: {e}")),
        }
    }

    if samples.is_empty() {
        return Err("No audio samples decoded".to_string());
    }

    let sample_rate = rate.unwrap_or(TARGET_SAMPLE_RATE as u32);
    if sample_rate == TARGET_SAMPLE_RATE as u32 {
        return Ok(samples);
    }
    let resampler = LinearResampler::create(sample_rate as i32, TARGET_SAMPLE_RATE)
        .ok_or_else(|| format!("Cannot create resampler {sample_rate} -> {TARGET_SAMPLE_RATE}"))?;
    Ok(resampler.resample(&samples, true))
}

/// Decode interleaved AudioBufferRef to mono f32 samples.
fn decode_to_mono_f32(
    decoded: &symphonia::core::audio::AudioBufferRef,
    num_channels: usize,
) -> Vec<f32> {
    use symphonia::core::audio::SampleBuffer;

    let mut buf = SampleBuffer::<f32>::new(decoded.capacity() as u64, *decoded.spec());
    buf.copy_interleaved_ref(decoded.clone());
    let all = buf.samples();

    if num_channels == 1 {
        return all.to_vec();
    }

    let num_frames = all.len() / num_channels;
    let mut mono = Vec::with_capacity(num_frames);
    for frame in 0..num_frames {
        let mut sum = 0.0f32;
        for ch in 0..num_channels {
            sum += all[frame * num_channels + ch];
        }
        mono.push(sum / num_channels as f32);
    }
    mono
}

/// Is ffmpeg on PATH (used as a decode fallback)?
fn ffmpeg_available() -> bool {
    Command::new("ffmpeg")
        .arg("-version")
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
        .is_ok()
}

/// Decode via ffmpeg to raw s16le 16 kHz mono PCM, then convert to f32.
/// Only the [start_sec, end_sec) range is decoded.
fn decode_with_ffmpeg(path: &str, start_sec: f32, end_sec: f32) -> Result<Vec<f32>, String> {
    let mut args: Vec<String> = vec!["-i".to_string(), path.to_string()];
    if start_sec > 0.0 {
        args.push("-ss".to_string());
        args.push(format!("{start_sec}"));
    }
    if end_sec.is_finite() {
        args.push("-t".to_string());
        args.push(format!("{}", (end_sec - start_sec).max(0.01)));
    }
    args.extend(
        [
            "-f",
            "s16le",
            "-acodec",
            "pcm_s16le",
            "-ac",
            "1",
            "-ar",
            &TARGET_SAMPLE_RATE.to_string(),
            "-",
        ]
        .iter()
        .map(|s| s.to_string()),
    );

    let mut child = Command::new("ffmpeg")
        .args(&args)
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .spawn()
        .map_err(|e| format!("Cannot start ffmpeg: {e}"))?;

    let mut raw = Vec::new();
    child
        .stdout
        .as_mut()
        .ok_or("ffmpeg has no stdout")?
        .read_to_end(&mut raw)
        .map_err(|e| format!("ffmpeg read failed: {e}"))?;
    let _ = child.wait();

    if raw.len() < 2 {
        return Err("ffmpeg produced no audio".to_string());
    }
    let samples = raw
        .chunks_exact(2)
        .map(|c| i16::from_le_bytes([c[0], c[1]]) as f32 / 32768.0)
        .collect();
    Ok(samples)
}

/// Pack float32 samples in [-1, 1] into a 16 kHz mono wav file in memory.
pub fn samples_to_wav(samples: &[f32], sample_rate: i32) -> Vec<u8> {
    let pcm: Vec<u8> = samples
        .iter()
        .map(|s| {
            let v = (s * 32767.0).clamp(-32768.0, 32767.0) as i16;
            v.to_le_bytes()
        })
        .flatten()
        .collect();

    let mut wav = Vec::with_capacity(44 + pcm.len());
    wav.extend_from_slice(b"RIFF");
    wav.extend_from_slice(&((36 + pcm.len()) as u32).to_le_bytes());
    wav.extend_from_slice(b"WAVE");
    wav.extend_from_slice(b"fmt ");
    wav.extend_from_slice(&16u32.to_le_bytes());
    wav.extend_from_slice(&1u16.to_le_bytes()); // PCM
    wav.extend_from_slice(&1u16.to_le_bytes()); // mono
    wav.extend_from_slice(&(sample_rate as u32).to_le_bytes());
    wav.extend_from_slice(&((sample_rate * 2) as u32).to_le_bytes()); // byte rate
    wav.extend_from_slice(&2u16.to_le_bytes()); // block align
    wav.extend_from_slice(&16u16.to_le_bytes()); // bits
    wav.extend_from_slice(b"data");
    wav.extend_from_slice(&(pcm.len() as u32).to_le_bytes());
    wav.extend_from_slice(&pcm);
    wav
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    /// Tests the 44-byte wav header and int16 packing.
    fn wav_header_is_valid() {
        let wav = samples_to_wav(&[0.0, 1.0, -1.0], 16000);
        assert_eq!(&wav[0..4], b"RIFF");
        assert_eq!(&wav[8..12], b"WAVE");
        assert_eq!(wav.len(), 44 + 6);
        // 44-byte header, then 3 mono int16 samples:
        // 0.0 -> 0, 1.0 -> 32767, -1.0 -> -32767 (same as the Python reference)
        assert_eq!(&wav[44..50], &[0x00, 0x00, 0xff, 0x7f, 0x01, 0x80]);
    }

    #[test]
    /// Tests that range decoding returns exactly that time slice of the file.
    fn decode_range_returns_exactly_that_slice() {
        // 2 s at 16 kHz: first half-second loud (0.25), 1 s silence,
        // last half-second louder (0.5).
        let sr = 16000usize;
        let mut samples = vec![0.0f32; sr * 2];
        for s in samples[..sr / 2].iter_mut() {
            *s = 0.25;
        }
        for s in samples[(sr * 3 / 2)..].iter_mut() {
            *s = 0.5;
        }
        let dir = std::env::temp_dir().join("doubao-range-test");
        std::fs::create_dir_all(&dir).unwrap();
        let wav = dir.join("tone.wav");
        std::fs::write(&wav, samples_to_wav(&samples, sr as i32)).unwrap();

        let all = decode_to_mono_16k(wav.to_str().unwrap(), 0.0, 0.0).unwrap();
        assert_eq!(all.len(), sr * 2, "whole file");

        // Range 1.0..2.0 s: silent first, loud tail.
        let part = decode_to_mono_16k(wav.to_str().unwrap(), 1.0, 2.0).unwrap();
        assert_eq!(part.len(), sr, "range length");
        assert!(part[0].abs() < 0.01, "range starts in silence, got {}", part[0]);
        assert!(
            (part[sr - 1] - 0.5).abs() < 0.01,
            "range ends loud, got {}",
            part[sr - 1]
        );

        // Range 0.0..0.5 s: the loud head only.
        let head = decode_to_mono_16k(wav.to_str().unwrap(), 0.0, 0.5).unwrap();
        assert_eq!(head.len(), sr / 2);
        assert!((head[0] - 0.25).abs() < 0.01, "got {}", head[0]);
    }
}
