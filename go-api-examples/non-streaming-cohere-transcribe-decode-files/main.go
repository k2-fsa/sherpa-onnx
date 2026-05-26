package main

import (
	"log"
	"strings"

	sherpa "github.com/k2-fsa/sherpa-onnx-go/sherpa_onnx"
)

func main() {
	log.SetFlags(log.LstdFlags | log.Lmicroseconds)

	config := sherpa.OfflineRecognizerConfig{}

	config.ModelConfig.CohereTranscribe.Encoder = "./sherpa-onnx-cohere-transcribe-14-lang-int8-2026-04-01/encoder.int8.onnx"
	config.ModelConfig.CohereTranscribe.Decoder = "./sherpa-onnx-cohere-transcribe-14-lang-int8-2026-04-01/decoder.int8.onnx"
	config.ModelConfig.CohereTranscribe.UsePunct = 1
	config.ModelConfig.CohereTranscribe.UseInverseTextNormalization = 1
	config.ModelConfig.Tokens = "./sherpa-onnx-cohere-transcribe-14-lang-int8-2026-04-01/tokens.txt"

	waveFilename := "./sherpa-onnx-cohere-transcribe-14-lang-int8-2026-04-01/test_wavs/en.wav"

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

	stream.SetOption("language", "en")
	stream.AcceptWaveform(wave.SampleRate, wave.Samples)

	recognizer.Decode(stream)
	log.Println("Decoding done!")
	result := stream.GetResult()

	log.Println("Text: " + strings.ToLower(result.Text))
}
