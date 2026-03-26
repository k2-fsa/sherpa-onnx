package main

import (
	"log"
	"strings"

	sherpa "github.com/k2-fsa/sherpa-onnx-go/sherpa_onnx"
)

func main() {
	log.SetFlags(log.LstdFlags | log.Lmicroseconds)

	config := sherpa.OfflineRecognizerConfig{}

	config.ModelConfig.Omnilingual.Model = "./sherpa-onnx-omnilingual-asr-1600-languages-300M-ctc-int8-2025-11-12/model.int8.onnx"
	config.ModelConfig.Tokens = "./sherpa-onnx-omnilingual-asr-1600-languages-300M-ctc-int8-2025-11-12/tokens.txt"

	waveFilename := "./sherpa-onnx-omnilingual-asr-1600-languages-300M-ctc-int8-2025-11-12/test_wavs/en.wav"

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
