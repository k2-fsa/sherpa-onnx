use clap::Parser;
use sherpa_onnx::{
    write, OfflineSpeechDenoiser, OfflineSpeechDenoiserConfig,
    OfflineSpeechDenoiserGtcrnModelConfig, Wave,
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

    let config = OfflineSpeechDenoiserConfig {
        model: sherpa_onnx::OfflineSpeechDenoiserModelConfig {
            gtcrn: OfflineSpeechDenoiserGtcrnModelConfig {
                model: Some(args.model),
            },
            ..Default::default()
        },
    };

    let denoiser = OfflineSpeechDenoiser::create(&config)
        .ok_or_else(|| anyhow::anyhow!("Failed to create offline GTCRN denoiser"))?;
    let wave =
        Wave::read(&args.input).ok_or_else(|| anyhow::anyhow!("Failed to read {}", args.input))?;

    let audio = denoiser.run(wave.samples(), wave.sample_rate());
    anyhow::ensure!(
        write(&args.output, &audio.samples, audio.sample_rate),
        "Failed to save {}",
        args.output
    );

    println!("Saved to {}", args.output);
    Ok(())
}
