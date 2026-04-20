package main

import (
	"log"
	"strings"

	sherpa "github.com/k2-fsa/sherpa-onnx-go/sherpa_onnx"
)

func main() {
	log.SetFlags(log.LstdFlags | log.Lmicroseconds)

	config := sherpa.OfflineRecognizerConfig{}

	config.ModelConfig.FunAsrNano.EncoderAdaptor = "./sherpa-onnx-funasr-nano-int8-2025-12-30/encoder_adaptor.int8.onnx"
	config.ModelConfig.FunAsrNano.LLM = "./sherpa-onnx-funasr-nano-int8-2025-12-30/llm.int8.onnx"
	config.ModelConfig.FunAsrNano.Embedding = "./sherpa-onnx-funasr-nano-int8-2025-12-30/embedding.int8.onnx"
	config.ModelConfig.FunAsrNano.Tokenizer = "./sherpa-onnx-funasr-nano-int8-2025-12-30/Qwen3-0.6B"
	// Seed for reproducibility (default: 42)
	config.ModelConfig.FunAsrNano.Seed = 42

	config.ModelConfig.Tokens = ""

	waveFilename := "./sherpa-onnx-funasr-nano-int8-2025-12-30/test_wavs/lyrics.wav"

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
