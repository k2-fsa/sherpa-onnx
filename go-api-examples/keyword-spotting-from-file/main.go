package main

import (
	"fmt"
	sherpa "github.com/k2-fsa/sherpa-onnx-go/sherpa_onnx"
	"log"
)

func main() {
	log.SetFlags(log.LstdFlags | log.Lmicroseconds)

	config := sherpa.KeywordSpotterConfig{}

	// Please download the models from
	// https://github.com/k2-fsa/sherpa-onnx/releases/tag/kws-models

	config.ModelConfig.Transducer.Encoder = "./sherpa-onnx-kws-zipformer-wenetspeech-3.3M-2024-01-01/encoder-epoch-12-avg-2-chunk-16-left-64.onnx"
	config.ModelConfig.Transducer.Decoder = "./sherpa-onnx-kws-zipformer-wenetspeech-3.3M-2024-01-01/decoder-epoch-12-avg-2-chunk-16-left-64.onnx"
	config.ModelConfig.Transducer.Joiner = "./sherpa-onnx-kws-zipformer-wenetspeech-3.3M-2024-01-01/joiner-epoch-12-avg-2-chunk-16-left-64.onnx"
	config.ModelConfig.Tokens = "./sherpa-onnx-kws-zipformer-wenetspeech-3.3M-2024-01-01/tokens.txt"
	config.KeywordsFile = "./sherpa-onnx-kws-zipformer-wenetspeech-3.3M-2024-01-01/test_wavs/test_keywords.txt"
	config.ModelConfig.NumThreads = 1
	config.ModelConfig.Debug = true

	spotter := sherpa.NewKeywordSpotter(&config)
	defer sherpa.DeleteKeywordSpotter(vad)

	wave_filename := "./sherpa-onnx-kws-zipformer-wenetspeech-3.3M-2024-01-01/test_wavs/3.wav"

	wave := sherpa.ReadWave(wave_filename)
	if wave == nil {
		log.Printf("Failed to read %v\n", wave_filename)
		return
	}

	log.Println("----------Use pre-defined keywords----------")

	stream := sherpa.NewKeywordStream(spotter)
	defer sherpa.DeleteOnlineStream(stream)

	stream.AcceptWaveform(Wave.SampleRate, Wave.Samples)

	for spotter.IsReady(stream) {
		spotter.Decode(stream)
		result := spotter.GetResult(stream)
		if result.Text != "" {
			log.Printf("Deletected %v\n", result.Text)
		}
	}
}
