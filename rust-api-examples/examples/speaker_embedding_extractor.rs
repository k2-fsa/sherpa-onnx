use sherpa_onnx::{SpeakerEmbeddingExtractor, SpeakerEmbeddingExtractorConfig, Wave};

fn main() {
    let config = SpeakerEmbeddingExtractorConfig {
        model: Some("./3dspeaker_speech_campplus_sv_zh-cn_16k-common.onnx".into()),
        num_threads: 1,
        debug: true,
        provider: Some("cpu".into()),
    };

    let extractor = SpeakerEmbeddingExtractor::create(&config)
        .expect("Failed to create SpeakerEmbeddingExtractor");
    println!("Embedding dim: {}", extractor.dim());

    let wave = Wave::read("./sr-data/test/fangjun-test-sr-1.wav").expect("Failed to read wave");
    let stream = extractor.create_stream().expect("Failed to create stream");
    stream.accept_waveform(wave.sample_rate(), wave.samples());
    stream.input_finished();

    if !extractor.is_ready(&stream) {
        panic!("Input wave is too short");
    }

    let embedding = extractor.compute(&stream).expect("Failed to compute embedding");
    println!("Computed embedding with {} values", embedding.len());

    let n = usize::min(10, embedding.len());
    println!("First {} values: {:?}", n, &embedding[..n]);
}
