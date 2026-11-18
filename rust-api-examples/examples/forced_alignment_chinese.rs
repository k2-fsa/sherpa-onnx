use anyhow::Result;
use clap::Parser;
use sherpa_onnx::{ForcedAligner, ForcedAlignerConfig, Wave};
use std::time::Instant;

#[derive(Debug, Parser)]
struct Args {
    /// Path to the model directory
    #[arg(long)]
    model_dir: String,

    /// Path to the test wav
    #[arg(long)]
    wav: String,

    /// Transcript of the wav; it must match the audio
    #[arg(long)]
    text: String,
}

fn main() -> Result<()> {
    let args = Args::parse();

    let wave = Wave::read(&args.wav).ok_or_else(|| anyhow::anyhow!("Failed to read WAV file"))?;

    let mut config = ForcedAlignerConfig::default();
    config.model_dir = Some(args.model_dir.clone());
    config.lexicon = Some(format!("{}/lexicon.txt", args.model_dir));

    let aligner =
        ForcedAligner::create(&config).ok_or_else(|| anyhow::anyhow!("Failed to create the aligner"))?;
    let stream = aligner.create_stream();
    stream.accept_waveform(wave.sample_rate(), wave.samples());

    // The transcript must match the wav; both are passed on the command
    // line so they cannot get out of sync.

    let start = Instant::now();
    let result = aligner
        .compute(&stream, &args.text)
        .ok_or_else(|| anyhow::anyhow!("Alignment failed"))?;
    let elapsed_seconds = start.elapsed().as_secs_f32();

    // tiers[0] is the "phones" tier; every interval has a phone symbol.
    let phones = &result.tiers[0];
    println!("phones:");
    for interval in &phones.intervals {
        println!(
            "{:.3} - {:.3}  {}",
            interval.start, interval.end, interval.text
        );
    }
    println!();

    // tiers[1] is the "words" tier; an empty text marks a silence gap
    // between words.
    let words = &result.tiers[1];
    println!("words:");
    for interval in &words.intervals {
        let label = if interval.text.is_empty() {
            "<silence>"
        } else {
            interval.text.as_str()
        };
        println!("{:.3} - {:.3}  {}", interval.start, interval.end, label);
    }
    println!();

    let output = "zh.TextGrid";
    std::fs::write(output, &result.text_grid)?;
    println!("Saved to {}", output);

    let audio_duration = wave.samples().len() as f32 / wave.sample_rate() as f32;
    let rtf = elapsed_seconds / audio_duration;

    println!("Elapsed seconds: {:.3}", elapsed_seconds);
    println!("Audio duration in seconds: {:.3}", audio_duration);
    println!("RTF: {:.3}/{:.3} = {:.3}", elapsed_seconds, audio_duration, rtf);

    Ok(())
}
