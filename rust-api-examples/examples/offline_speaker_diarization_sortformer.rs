// Speaker diarization with Nemotron-3-Diarization, an end-to-end streaming
// Sortformer model. It needs neither a speaker embedding model nor
// clustering.
//
// wget https://github.com/k2-fsa/sherpa-onnx/releases/download/speaker-segmentation-models/sherpa-onnx-nemotron-3-diarization.tar.bz2
// tar xvf sherpa-onnx-nemotron-3-diarization.tar.bz2
// wget https://github.com/k2-fsa/sherpa-onnx/releases/download/speaker-segmentation-models/0-four-speakers-zh.wav
//
// cargo run --example offline_speaker_diarization_sortformer

use sherpa_onnx::{
    OfflineSpeakerDiarization, OfflineSpeakerDiarizationConfig,
    OfflineSpeakerSegmentationModelConfig, OfflineSpeakerSegmentationSortformerModelConfig, Wave,
};

fn main() {
    let config = OfflineSpeakerDiarizationConfig {
        segmentation: OfflineSpeakerSegmentationModelConfig {
            sortformer: OfflineSpeakerSegmentationSortformerModelConfig {
                model: Some("./sherpa-onnx-nemotron-3-diarization/model.int8.onnx".into()),
                ..Default::default()
            },
            num_threads: 2,
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
