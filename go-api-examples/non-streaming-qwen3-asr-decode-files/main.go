package main

import (
	"log"
	"strings"

	sherpa "github.com/k2-fsa/sherpa-onnx-go/sherpa_onnx"
)

func main() {
	log.SetFlags(log.LstdFlags | log.Lmicroseconds)

	config := sherpa.OfflineRecognizerConfig{}

	config.ModelConfig.Qwen3ASR.ConvFrontend = "./sherpa-onnx-qwen3-asr-0.6B-int8-2026-03-25/conv_frontend.onnx"
	config.ModelConfig.Qwen3ASR.Encoder = "./sherpa-onnx-qwen3-asr-0.6B-int8-2026-03-25/encoder.int8.onnx"
	config.ModelConfig.Qwen3ASR.Decoder = "./sherpa-onnx-qwen3-asr-0.6B-int8-2026-03-25/decoder.int8.onnx"
	config.ModelConfig.Qwen3ASR.Tokenizer = "./sherpa-onnx-qwen3-asr-0.6B-int8-2026-03-25/tokenizer"
	config.ModelConfig.Qwen3ASR.Hotwords = ""
	// Seed for reproducibility (default: 42)
	config.ModelConfig.Qwen3ASR.Seed = 42

	config.ModelConfig.Tokens = ""

	waveFilename := "./sherpa-onnx-qwen3-asr-0.6B-int8-2026-03-25/test_wavs/raokouling.wav"

	wave := sherpa.ReadWave(waveFilename)
	if wave == nil {
		log.Fatalf("Failed to read %v", waveFilename)
	}

	log.Println("Initializing recognizer (may take several seconds)")
	recognizer := sherpa.NewOfflineRecognizer(&config)
	log.Println("Recognizer created!")
	defer sherpa.DeleteOfflineRecognizer(recognizer)

	log.Println("Start decoding!")
	stream := sherpa.NewOfflineStream(recognizer)
	defer sherpa.DeleteOfflineStream(stream)

	stream.AcceptWaveform(wave.SampleRate, wave.Samples)

	recognizer.Decode(stream)
	log.Println("Decoding done!")
	result := stream.GetResult()

	log.Println("Text: " + strings.ToLower(result.Text))
}
