use clap::Parser;
use sherpa_onnx::{
    write, OfflineSpeechDenoiserGtcrnModelConfig, OnlineSpeechDenoiser, OnlineSpeechDenoiserConfig,
    Wave,
};

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    #[arg(long)]
    model: String,

    #[arg(long)]
    input: String,

    #[arg(long)]
    output: String,
}

fn main() -> anyhow::Result<()> {
    let args = Args::parse();

    let config = OnlineSpeechDenoiserConfig {
        model: sherpa_onnx::OfflineSpeechDenoiserModelConfig {
            gtcrn: OfflineSpeechDenoiserGtcrnModelConfig {
                model: Some(args.model),
            },
            ..Default::default()
        },
    };

    let denoiser = OnlineSpeechDenoiser::create(&config)
        .ok_or_else(|| anyhow::anyhow!("Failed to create streaming GTCRN denoiser"))?;
    let wave =
        Wave::read(&args.input).ok_or_else(|| anyhow::anyhow!("Failed to read {}", args.input))?;

    let frame_shift = denoiser.frame_shift_in_samples() as usize;
    let mut enhanced = Vec::new();

    for chunk in wave.samples().chunks(frame_shift.max(1)) {
        let audio = denoiser.run(chunk, wave.sample_rate());
        enhanced.extend_from_slice(&audio.samples);
    }

    enhanced.extend_from_slice(&denoiser.flush().samples);

    anyhow::ensure!(
        write(&args.output, &enhanced, denoiser.sample_rate()),
        "Failed to save {}",
        args.output
    );

    println!("Saved to {}", args.output);
    Ok(())
}
