package main

import (
	"bytes"
	"encoding/binary"
	sherpa "github.com/k2-fsa/sherpa-onnx-go/sherpa_onnx"
	"github.com/youpy/go-wav"
	"log"
	"os"
	"strings"
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

	wav_filename := "./sherpa-onnx-streaming-zipformer-ctc-small-2024-03-18/test_wavs/8k.wav"

	samples, sampleRate := readWave(wav_filename)

	log.Println("Initializing recognizer (may take several seconds)")
	recognizer := sherpa.NewOnlineRecognizer(&config)
	log.Println("Recognizer created!")
	defer sherpa.DeleteOnlineRecognizer(recognizer)

	log.Println("Start decoding!")
	stream := sherpa.NewOnlineStream(recognizer)
	defer sherpa.DeleteOnlineStream(stream)

	stream.AcceptWaveform(sampleRate, samples)

	tailPadding := make([]float32, int(float32(sampleRate)*0.3))
	stream.AcceptWaveform(sampleRate, tailPadding)

	for recognizer.IsReady(stream) {
		recognizer.Decode(stream)
	}
	log.Println("Decoding done!")
	result := recognizer.GetResult(stream)
	log.Println(strings.ToLower(result.Text))
	log.Printf("Wave duration: %v seconds", float32(len(samples))/float32(sampleRate))
}

func readWave(filename string) (samples []float32, sampleRate int) {
	file, _ := os.Open(filename)
	defer file.Close()

	reader := wav.NewReader(file)
	format, err := reader.Format()
	if err != nil {
		log.Fatalf("Failed to read wave format")
	}

	if format.AudioFormat != 1 {
		log.Fatalf("Support only PCM format. Given: %v\n", format.AudioFormat)
	}

	if format.NumChannels != 1 {
		log.Fatalf("Support only 1 channel wave file. Given: %v\n", format.NumChannels)
	}

	if format.BitsPerSample != 16 {
		log.Fatalf("Support only 16-bit per sample. Given: %v\n", format.BitsPerSample)
	}

	reader.Duration() // so that it initializes reader.Size

	buf := make([]byte, reader.Size)
	n, err := reader.Read(buf)
	if n != int(reader.Size) {
		log.Fatalf("Failed to read %v bytes. Returned %v bytes\n", reader.Size, n)
	}

	samples = samplesInt16ToFloat(buf)
	sampleRate = int(format.SampleRate)

	return
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
