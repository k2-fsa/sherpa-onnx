use sherpa_onnx::{
    SpeakerEmbeddingExtractor, SpeakerEmbeddingExtractorConfig, SpeakerEmbeddingManager, Wave,
};

fn compute_embedding(extractor: &SpeakerEmbeddingExtractor, filename: &str) -> Vec<f32> {
    let wave = Wave::read(filename).unwrap_or_else(|| panic!("Failed to read {}", filename));
    let stream = extractor.create_stream().expect("Failed to create stream");
    stream.accept_waveform(wave.sample_rate(), wave.samples());
    stream.input_finished();

    if !extractor.is_ready(&stream) {
        panic!("The input wave file {} is too short!", filename);
    }

    extractor
        .compute(&stream)
        .unwrap_or_else(|| panic!("Failed to compute embedding for {}", filename))
}

fn main() {
    let config = SpeakerEmbeddingExtractorConfig {
        model: Some("./3dspeaker_speech_campplus_sv_zh-cn_16k-common.onnx".into()),
        num_threads: 1,
        debug: true,
        provider: Some("cpu".into()),
    };

    let extractor = SpeakerEmbeddingExtractor::create(&config)
        .expect("Failed to create SpeakerEmbeddingExtractor");
    let manager = SpeakerEmbeddingManager::create(extractor.dim())
        .expect("Failed to create SpeakerEmbeddingManager");

    let spk1 = vec![
        compute_embedding(&extractor, "./sr-data/enroll/fangjun-sr-1.wav"),
        compute_embedding(&extractor, "./sr-data/enroll/fangjun-sr-2.wav"),
        compute_embedding(&extractor, "./sr-data/enroll/fangjun-sr-3.wav"),
    ];
    let spk2 = vec![
        compute_embedding(&extractor, "./sr-data/enroll/leijun-sr-1.wav"),
        compute_embedding(&extractor, "./sr-data/enroll/leijun-sr-2.wav"),
    ];

    assert!(manager.add_list("fangjun", &spk1));
    assert!(manager.contains("fangjun"));

    let flattened_spk2: Vec<f32> = spk2.iter().flat_map(|v| v.iter().copied()).collect();
    assert!(manager.add_list_flattened("leijun", &flattened_spk2));
    assert!(manager.contains("leijun"));
    assert_eq!(manager.num_speakers(), 2);

    println!("Registered speakers: {:?}", manager.get_all_speakers());

    let v1 = compute_embedding(&extractor, "./sr-data/test/fangjun-test-sr-1.wav");
    let v2 = compute_embedding(&extractor, "./sr-data/test/leijun-test-sr-1.wav");
    let v3 = compute_embedding(&extractor, "./sr-data/test/liudehua-test-sr-1.wav");

    let threshold = 0.6;

    println!(
        "fangjun-test-sr-1.wav => {}",
        manager.search(&v1, threshold).unwrap_or_else(|| "unknown".to_string())
    );
    println!(
        "leijun-test-sr-1.wav => {}",
        manager.search(&v2, threshold).unwrap_or_else(|| "unknown".to_string())
    );
    println!(
        "liudehua-test-sr-1.wav => {}",
        manager.search(&v3, threshold).unwrap_or_else(|| "unknown".to_string())
    );

    let best_matches = manager.get_best_matches(&v1, threshold, 2);
    println!("Best matches for fangjun-test-sr-1.wav: {:?}", best_matches);

    println!("fangjun verification for v1: {}", manager.verify("fangjun", &v1, threshold));
    println!("fangjun verification for v2: {}", manager.verify("fangjun", &v2, threshold));

    assert!(manager.remove("fangjun"));
    println!("After removing fangjun: {:?}", manager.get_all_speakers());

    assert!(manager.remove("leijun"));
    println!("After removing leijun: {:?}", manager.get_all_speakers());
}
