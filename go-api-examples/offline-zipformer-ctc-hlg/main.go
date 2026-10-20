// go-api-examples/offline-zipformer-ctc-hlg/main.go
//
// Copyright (c)  2026  Xiaomi Corporation
//
// Offline CTC decoding with an HLG graph via CtcFstDecoderConfig.
//
// wget https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-zipformer-ctc-en-2023-10-02.tar.bz2
// tar xvf sherpa-onnx-zipformer-ctc-en-2023-10-02.tar.bz2
// rm sherpa-onnx-zipformer-ctc-en-2023-10-02.tar.bz2
package main

import (
	"flag"
	"fmt"

	sherpa "github.com/k2-fsa/sherpa-onnx-go/sherpa_onnx"
)

func main() {
	var config sherpa.OfflineRecognizerConfig

	flag.StringVar(&config.ModelConfig.ZipformerCtc.Model, "zipformer-ctc", "", "Path to the Zipformer CTC model")
	flag.StringVar(&config.ModelConfig.Tokens, "tokens", "", "Path to tokens.txt")

	// This is what the example exercises: the graph and max-active reach the
	// C API through CtcFstDecoderConfig.
	flag.StringVar(&config.CtcFstDecoderConfig.Graph, "graph", "", "Path to H.fst, HL.fst or HLG.fst")
	flag.IntVar(&config.CtcFstDecoderConfig.MaxActive, "max-active", 3000, "Decoder max-active setting")

	flag.Parse()

	if len(flag.Args()) != 1 {
		panic("Usage: ./offline-zipformer-ctc-hlg --zipformer-ctc <model> --tokens <tokens> --graph <fst> <wav>")
	}
	waveFilename := flag.Arg(0)

	config.ModelConfig.Debug = 0
	config.ModelConfig.Provider = "cpu"
	config.ModelConfig.NumThreads = 1

	wave := sherpa.ReadWave(waveFilename)

	recognizer := sherpa.NewOfflineRecognizer(&config)
	defer sherpa.DeleteOfflineRecognizer(recognizer)
	stream := sherpa.NewOfflineStream(recognizer)
	defer sherpa.DeleteOfflineStream(stream)
	stream.AcceptWaveform(wave.SampleRate, wave.Samples)
	recognizer.Decode(stream)

	result := stream.GetResult()
	fmt.Printf("text: %s\n", result.Text)
}
