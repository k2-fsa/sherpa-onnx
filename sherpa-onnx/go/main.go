package main

import (
	"bytes"
	"encoding/binary"
	"fmt"
	"github.com/DylanMeeus/GoAudio/wave"
	"log"
)

func main() {
	fmt.Println("hello 2")
	config := OnlineRecognizerConfig{}
	config.FeatConfig = FeatureConfig{SampleRate: 16000, FeatureDim: 80}
	config.ModelConfig = OnlineTransducerModelConfig{
		Encoder:    "./sherpa-onnx-streaming-zipformer-en-2023-06-26/encoder-epoch-99-avg-1-chunk-16-left-128.onnx",
		Decoder:    "./sherpa-onnx-streaming-zipformer-en-2023-06-26/decoder-epoch-99-avg-1-chunk-16-left-128.onnx",
		Joiner:     "./sherpa-onnx-streaming-zipformer-en-2023-06-26/joiner-epoch-99-avg-1-chunk-16-left-128.onnx",
		Tokens:     "./sherpa-onnx-streaming-zipformer-en-2023-06-26/tokens.txt",
		NumThreads: 1,
		Provider:   "cpu",
		Debug:      1,
		ModelType:  "zipformer2",
	}
	config.DecodingMethod = "greedy_search"
	config.EnableEndpoint = 0

	recognizer := NewOnlineRecognizer(&config)
	defer DeleteOnlineRecognizer(recognizer)

	testFile := "./sherpa-onnx-streaming-zipformer-en-2023-06-26/test_wavs/0.wav"
	wav, err := wave.ReadWaveFile(testFile)
	if err != nil {
		log.Fatalf("Failed to read wave file: %s, %v", testFile, err)
	}
	log.Printf("Sample rate: %v", wav.SampleRate)
	log.Printf("Bits per sample: %v", wav.BitsPerSample)

	if wav.NumChannels != 1 {
		log.Fatalf("Only single channel is supported. Given: %v", wav.NumChannels)
	}

	if wav.BitsPerSample != 16 {
		log.Fatalf("Only 16-bit per sample is supported. Given: %v", wav.BitsPerSample)
	}

	samplesf32 := samplesInt16ToFloat(wav.RawData)

	stream := NewOnlineStream(recognizer)
	defer DeleteOnlineStream(stream)

	stream.AcceptWaveform(wav.SampleRate, samplesf32)

	tailPadding := make([]float32, int(float32(wav.SampleRate)*0.3))
	stream.AcceptWaveform(wav.SampleRate, tailPadding)

	for recognizer.IsReady(stream) {
		recognizer.Decode(stream)
	}
	result := recognizer.GetResult(stream)
	log.Println(result)
}

func samplesInt16ToFloat(inSamples []byte) []float32 {
	numSamples := len(inSamples) / 2
	outSamples := make([]float32, numSamples)

	for i := 0; i != numSamples; i++ {
		s := inSamples[i*2 : (i+1)*2]

		var s16 int16
		buf := bytes.NewReader(s)
		err := binary.Read(buf, binary.LittleEndian, &s16)
		if err != nil {
			log.Fatal("Failed to parse 16-bit sample")
		}
		outSamples[i] = float32(s16) / 32768
	}

	return outSamples
}
