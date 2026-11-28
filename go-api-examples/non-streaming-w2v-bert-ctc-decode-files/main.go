package main

import (
	"log"
	"strings"

	sherpa "github.com/k2-fsa/sherpa-onnx-go/v2/sherpa_onnx"
)

func main() {
	log.SetFlags(log.LstdFlags | log.Lmicroseconds)

	config := sherpa.OfflineRecognizerConfig{}

	config.ModelConfig.W2vBertCtc.Model = "./sherpa-onnx-w2v-bert-ctc-gl-int8-2026-11-20/model.int8.onnx"
	config.ModelConfig.Tokens = "./sherpa-onnx-w2v-bert-ctc-gl-int8-2026-11-20/tokens.txt"

	waveFilename := "./sherpa-onnx-w2v-bert-ctc-gl-int8-2026-11-20/test_wavs/0.wav"

	wave := sherpa.ReadWave(waveFilename)
	if wave == nil {
		log.Fatalf("Failed to read %v", waveFilename)
	}

	log.Println("Initializing recognizer (may take several seconds)")
	recognizer := sherpa.NewOfflineRecognizer(&config)
	if recognizer == nil {
		log.Fatalf("Failed to create the recognizer; check the model paths")
	}
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
