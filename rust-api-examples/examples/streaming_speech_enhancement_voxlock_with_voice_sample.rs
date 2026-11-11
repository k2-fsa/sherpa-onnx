// VoxLock is a target-speaker voice isolation model. It keeps the voice of one
// speaker, e.g., the caller in a phone call, and removes the other voices.
//
// The model is from https://github.com/arbazsiddiqui/voxlock
//
// This example provides a voice sample of the speaker to keep, so VoxLock
// isolates that speaker from the first sample. The voice sample must have at
// least 3 seconds of the speaker's voice; from a longer file, VoxLock uses the
// loudest 3 seconds within its first 30 seconds. See
// streaming_speech_enhancement_voxlock.rs if you don't have a voice sample.
//
// Please download the model from
// https://github.com/k2-fsa/sherpa-onnx/releases/tag/speech-enhancement-models
use clap::Parser;
use sherpa_onnx::{
    write, OnlineSpeechDenoiser, OnlineSpeechDenoiserConfig,
    OfflineSpeechDenoiserVoxLockModelConfig, Wave,
};

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    #[arg(long)]
    enroll_model: String,

    #[arg(long)]
    chunk_model: String,

    #[arg(long)]
    voice_sample: String,

    #[arg(long)]
    input: String,

    #[arg(long)]
    output: String,
}

fn main() -> anyhow::Result<()> {
    let args = Args::parse();

    let config = OnlineSpeechDenoiserConfig {
        model: sherpa_onnx::OfflineSpeechDenoiserModelConfig {
            voxlock: OfflineSpeechDenoiserVoxLockModelConfig {
                enroll_model: Some(args.enroll_model),
                chunk_model: Some(args.chunk_model),
                voice_sample: Some(args.voice_sample),
            },
            ..Default::default()
        },
    };

    let denoiser = OnlineSpeechDenoiser::create(&config)
        .ok_or_else(|| anyhow::anyhow!("Failed to create streaming VoxLock denoiser"))?;
    let wave =
        Wave::read(&args.input).ok_or_else(|| anyhow::anyhow!("Failed to read {}", args.input))?;

    let frame_shift = denoiser.frame_shift_in_samples() as usize;
    let mut enhanced = Vec::new();

    let begin = std::time::Instant::now();
    for chunk in wave.samples().chunks(frame_shift.max(1)) {
        let audio = denoiser.run(chunk, wave.sample_rate());
        enhanced.extend_from_slice(&audio.samples);
    }

    enhanced.extend_from_slice(&denoiser.flush().samples);
    let elapsed_seconds = begin.elapsed().as_secs_f64();

    anyhow::ensure!(
        write(&args.output, &enhanced, denoiser.sample_rate()),
        "Failed to save {}",
        args.output
    );

    let duration = wave.samples().len() as f64 / wave.sample_rate() as f64;
    println!("Saved to {}", args.output);
    println!("Elapsed seconds: {:.3}", elapsed_seconds);
    println!("Audio duration in seconds: {:.3}", duration);
    println!(
        "RTF: {:.3}/{:.3} = {:.3}",
        elapsed_seconds,
        duration,
        elapsed_seconds / duration
    );
    Ok(())
}
