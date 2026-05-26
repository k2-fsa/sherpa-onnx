use sherpa_onnx::{
    FastClusteringConfig, OfflineSpeakerDiarization, OfflineSpeakerDiarizationConfig,
    OfflineSpeakerSegmentationModelConfig, OfflineSpeakerSegmentationPyannoteModelConfig,
    SpeakerEmbeddingExtractorConfig, Wave,
};

fn main() {
    let config = OfflineSpeakerDiarizationConfig {
        segmentation: OfflineSpeakerSegmentationModelConfig {
            pyannote: OfflineSpeakerSegmentationPyannoteModelConfig {
                model: Some("./sherpa-onnx-pyannote-segmentation-3-0/model.onnx".into()),
            },
            ..Default::default()
        },
        embedding: SpeakerEmbeddingExtractorConfig {
            model: Some("./3dspeaker_speech_eres2net_base_sv_zh-cn_3dspeaker_16k.onnx".into()),
            ..Default::default()
        },
        clustering: FastClusteringConfig {
            num_clusters: 4,
            ..Default::default()
        },
        ..Default::default()
    };

    let sd = OfflineSpeakerDiarization::create(&config)
        .expect("Failed to initialize offline speaker diarization");

    let wave = Wave::read("./0-four-speakers-zh.wav").expect("Failed to read wave");

    assert_eq!(
        sd.sample_rate(),
        wave.sample_rate(),
        "Unexpected sample rate"
    );

    let result = sd
        .process(wave.samples())
        .expect("Failed to do speaker diarization");
    println!("Number of speakers: {}", result.num_speakers());
    println!("Number of segments: {}", result.num_segments());

    for s in result.sort_by_start_time() {
        println!("{:.3} -- {:.3} speaker_{:02}", s.start, s.end, s.speaker);
    }
}
