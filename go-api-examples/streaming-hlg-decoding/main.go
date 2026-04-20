package main

import (
	"log"
	"strings"

	sherpa "github.com/k2-fsa/sherpa-onnx-go/sherpa_onnx"
)

func main() {
	log.SetFlags(log.LstdFlags | log.Lmicroseconds)

	config := sherpa.OnlineRecognizerConfig{}
	config.FeatConfig = sherpa.FeatureConfig{SampleRate: 16000, FeatureDim: 80}

	// please download model files from
	// https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-streaming-zipformer-ctc-small-2024-03-18.tar.bz2
	config.ModelConfig.Zipformer2Ctc.Model = "./sherpa-onnx-streaming-zipformer-ctc-small-2024-03-18/ctc-epoch-30-avg-3-chunk-16-left-128.int8.onnx"
	config.ModelConfig.Tokens = "./sherpa-onnx-streaming-zipformer-ctc-small-2024-03-18/tokens.txt"

	config.ModelConfig.NumThreads = 1
	config.ModelConfig.Debug = 0
	config.ModelConfig.Provider = "cpu"
	config.CtcFstDecoderConfig.Graph = "./sherpa-onnx-streaming-zipformer-ctc-small-2024-03-18/HLG.fst"

	waveFilename := "./sherpa-onnx-streaming-zipformer-ctc-small-2024-03-18/test_wavs/8k.wav"

	wave := sherpa.ReadWave(waveFilename)
	if wave == nil {
		log.Fatalf("Failed to read %v", waveFilename)
	}

	log.Println("Initializing recognizer (may take several seconds)")
	recognizer := sherpa.NewOnlineRecognizer(&config)
	log.Println("Recognizer created!")
	defer sherpa.DeleteOnlineRecognizer(recognizer)

	log.Println("Start decoding!")
	stream := sherpa.NewOnlineStream(recognizer)
	defer sherpa.DeleteOnlineStream(stream)

	stream.AcceptWaveform(wave.SampleRate, wave.Samples)

	tailPadding := make([]float32, int(float32(wave.SampleRate)*0.3))
	stream.AcceptWaveform(wave.SampleRate, tailPadding)

	for recognizer.IsReady(stream) {
		recognizer.Decode(stream)
	}
	log.Println("Decoding done!")
	result := recognizer.GetResult(stream)
	log.Println(strings.ToLower(result.Text))
	log.Printf("Wave duration: %v seconds", float32(len(wave.Samples))/float32(wave.SampleRate))
}
