// rust-api-examples/examples/zipformer_ctc_hlg.rs
//
// Copyright (c)  2026  Xiaomi Corporation
//
// Offline CTC decoding with an HLG graph via ctc_fst_decoder_config.
//
// clang-format off
//
// wget https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-zipformer-ctc-en-2023-10-02.tar.bz2
// tar xvf sherpa-onnx-zipformer-ctc-en-2023-10-02.tar.bz2
// rm sherpa-onnx-zipformer-ctc-en-2023-10-02.tar.bz2
//
// cargo run --release --example zipformer_ctc_hlg -- \
//   --model ./sherpa-onnx-zipformer-ctc-en-2023-10-02/model.onnx \
//   --tokens ./sherpa-onnx-zipformer-ctc-en-2023-10-02/tokens.txt \
//   --graph ./sherpa-onnx-zipformer-ctc-en-2023-10-02/HLG.fst \
//   --wav ./sherpa-onnx-zipformer-ctc-en-2023-10-02/test_wavs/0.wav
//
// clang-format on

use clap::Parser;
use sherpa_onnx::{
  OfflineCtcFstDecoderConfig, OfflineRecognizer, OfflineRecognizerConfig, OfflineZipformerCtcModelConfig,
  Wave,
};

#[derive(Parser, Debug)]
struct Args {
  /// Path to the zipformer CTC model
  #[arg(long)]
  model: String,

  /// Path to tokens.txt
  #[arg(long)]
  tokens: String,

  /// Path to the decoding graph (H.fst, HL.fst or HLG.fst)
  #[arg(long)]
  graph: String,

  /// Path to a wav file
  #[arg(long)]
  wav: String,

  /// Number of threads
  #[arg(long, default_value_t = 1)]
  num_threads: i32,

  /// Enable debug logs
  #[arg(long, default_value_t = false)]
  debug: bool,
}

fn main() {
  let args = Args::parse();

  let mut config = OfflineRecognizerConfig::default();
  config.model_config.zipformer_ctc = OfflineZipformerCtcModelConfig {
    model: Some(args.model),
  };
  config.model_config.tokens = Some(args.tokens);
  config.model_config.num_threads = args.num_threads;
  config.model_config.debug = args.debug;

  // This is what the example exercises: the graph and max_active reach the C
  // API through OfflineCtcFstDecoderConfig.
  config.ctc_fst_decoder_config = OfflineCtcFstDecoderConfig {
    graph: Some(args.graph),
    max_active: 3000,
  };

  let recognizer =
    OfflineRecognizer::create(&config).expect("Failed to create OfflineRecognizer");

  let wave = Wave::read(&args.wav).expect("Failed to read WAV file");
  let stream = recognizer.create_stream();
  stream.accept_waveform(wave.sample_rate(), wave.samples());
  recognizer.decode(&stream);

  if let Some(result) = stream.get_result() {
    println!("Decoded text: {}", result.text);
  } else {
    eprintln!("No result produced");
  }
}
