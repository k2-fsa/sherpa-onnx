package main

import (
	"log"
	"strings"

	sherpa "github.com/k2-fsa/sherpa-onnx-go/sherpa_onnx"
)

func main() {
	log.SetFlags(log.LstdFlags | log.Lmicroseconds)

	config := sherpa.OfflineRecognizerConfig{}

	config.ModelConfig.Canary.Encoder = "./sherpa-onnx-nemo-canary-180m-flash-en-es-de-fr-int8/encoder.int8.onnx"
	config.ModelConfig.Canary.Decoder = "./sherpa-onnx-nemo-canary-180m-flash-en-es-de-fr-int8/decoder.int8.onnx"
	config.ModelConfig.Canary.SrcLang = "en"
	config.ModelConfig.Canary.TgtLang = "en"
	config.ModelConfig.Canary.UsePnc = 1
	config.ModelConfig.Tokens = "./sherpa-onnx-nemo-canary-180m-flash-en-es-de-fr-int8/tokens.txt"

	waveFilename := "./sherpa-onnx-nemo-canary-180m-flash-en-es-de-fr-int8/test_wavs/en.wav"

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

	log.Println("Text in English: " + strings.ToLower(result.Text))

	s := sherpa.NewOfflineStream(recognizer)
	defer sherpa.DeleteOfflineStream(s)

	s.AcceptWaveform(wave.SampleRate, wave.Samples)

	config.ModelConfig.Canary.TgtLang = "de"
	recognizer.SetConfig(&config)
	recognizer.Decode(s)
	result = s.GetResult()

	log.Println("Text in German: " + strings.ToLower(result.Text))
}
