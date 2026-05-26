package main

import (
	"log"
	"strings"

	sherpa "github.com/k2-fsa/sherpa-onnx-go/sherpa_onnx"
)

func main() {
	log.SetFlags(log.LstdFlags | log.Lmicroseconds)

	config := sherpa.OfflineRecognizerConfig{}

	config.ModelConfig.Moonshine.Encoder = "./sherpa-onnx-moonshine-tiny-en-quantized-2026-02-27/encoder_model.ort"
	config.ModelConfig.Moonshine.MergedDecoder = "./sherpa-onnx-moonshine-tiny-en-quantized-2026-02-27/decoder_model_merged.ort"
	config.ModelConfig.Tokens = "./sherpa-onnx-moonshine-tiny-en-quantized-2026-02-27/tokens.txt"

	waveFilename := "./sherpa-onnx-moonshine-tiny-en-quantized-2026-02-27/test_wavs/0.wav"

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
