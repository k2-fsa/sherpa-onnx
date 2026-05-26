use sherpa_onnx::{SpeakerEmbeddingExtractor, SpeakerEmbeddingExtractorConfig, Wave};

fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len(), "Vectors must have the same length");

    let mut dot = 0.0_f32;
    let mut sum_a = 0.0_f32;
    let mut sum_b = 0.0_f32;

    for (&x, &y) in a.iter().zip(b.iter()) {
        dot += x * y;
        sum_a += x * x;
        sum_b += y * y;
    }

    let mag_a = sum_a.sqrt();
    let mag_b = sum_b.sqrt();
    if mag_a > 0.0 && mag_b > 0.0 {
        dot / (mag_a * mag_b)
    } else {
        0.0
    }
}

fn compute_embedding(extractor: &SpeakerEmbeddingExtractor, wave_filename: &str) -> Vec<f32> {
    let wave = Wave::read(wave_filename)
        .unwrap_or_else(|| panic!("Failed to read {}", wave_filename));
    let stream = extractor.create_stream().expect("Failed to create stream");
    stream.accept_waveform(wave.sample_rate(), wave.samples());
    stream.input_finished();

    if !extractor.is_ready(&stream) {
        panic!("{} is too short", wave_filename);
    }

    extractor
        .compute(&stream)
        .unwrap_or_else(|| panic!("Failed to compute embedding for {}", wave_filename))
}

fn main() {
    let config = SpeakerEmbeddingExtractorConfig {
        model: Some("./wespeaker_zh_cnceleb_resnet34.onnx".into()),
        num_threads: 1,
        debug: true,
        provider: Some("cpu".into()),
    };

    let extractor = SpeakerEmbeddingExtractor::create(&config)
        .expect("Failed to create SpeakerEmbeddingExtractor");

    let embedding1 = compute_embedding(&extractor, "./fangjun-sr-1.wav");
    let embedding2 = compute_embedding(&extractor, "./fangjun-sr-2.wav");
    let embedding3 = compute_embedding(&extractor, "./leijun-sr-1.wav");

    let score12 = cosine_similarity(&embedding1, &embedding2);
    let score13 = cosine_similarity(&embedding1, &embedding3);
    let score23 = cosine_similarity(&embedding2, &embedding3);

    println!("Score between spk1 and spk2: {}", score12);
    println!("Score between spk1 and spk3: {}", score13);
    println!("Score between spk2 and spk3: {}", score23);
}
