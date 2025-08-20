package main

import (
	"bytes"
	"encoding/binary"
	"log"
	"os"
	"strings"

	sherpa "github.com/k2-fsa/sherpa-onnx-go/sherpa_onnx"
	"github.com/youpy/go-wav"
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

	samples, sampleRate := readWave(waveFilename)

	log.Println("Initializing recognizer (may take several seconds)")
	recognizer := sherpa.NewOfflineRecognizer(&config)
	log.Println("Recognizer created!")
	defer sherpa.DeleteOfflineRecognizer(recognizer)

	log.Println("Start decoding!")
	stream := sherpa.NewOfflineStream(recognizer)
	defer sherpa.DeleteOfflineStream(stream)

	stream.AcceptWaveform(sampleRate, samples)

	recognizer.Decode(stream)
	log.Println("Decoding done!")
	result := stream.GetResult()

	log.Println("Text in English: " + strings.ToLower(result.Text))

	s := sherpa.NewOfflineStream(recognizer)
	defer sherpa.DeleteOfflineStream(s)

	s.AcceptWaveform(sampleRate, samples)

	config.ModelConfig.Canary.TgtLang = "de"
	recognizer.SetConfig(&config)
	recognizer.Decode(s)
	result = s.GetResult()

	log.Println("Text in German: " + strings.ToLower(result.Text))
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
