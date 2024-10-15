package main

import (
	"fmt"
	portaudio "github.com/csukuangfj/portaudio-go"
	sherpa "github.com/k2-fsa/sherpa-onnx-go/sherpa_onnx"
	"log"
	"strings"
)

func main() {
	log.SetFlags(log.LstdFlags | log.Lmicroseconds)

	// 1. Create VAD
	config := sherpa.VadModelConfig{}

	// Please download silero_vad.onnx from
	// https://github.com/snakers4/silero-vad/raw/master/src/silero_vad/data/silero_vad.onnx

	config.SileroVad.Model = "./silero_vad.onnx"
	config.SileroVad.Threshold = 0.5
	config.SileroVad.MinSilenceDuration = 0.5
	config.SileroVad.MinSpeechDuration = 0.25
	config.SileroVad.WindowSize = 512
	config.SileroVad.MaxSpeechDuration = 5.0
	config.SampleRate = 16000
	config.NumThreads = 1
	config.Provider = "cpu"
	config.Debug = 1

	var bufferSizeInSeconds float32 = 20

	vad := sherpa.NewVoiceActivityDetector(&config, bufferSizeInSeconds)
	defer sherpa.DeleteVoiceActivityDetector(vad)

	// 2. Create ASR recognizer

	c := sherpa.OfflineRecognizerConfig{}
	c.FeatConfig.SampleRate = 16000
	c.FeatConfig.FeatureDim = 80
	c.ModelConfig.Whisper.Encoder = "./sherpa-onnx-whisper-tiny.en/tiny.en-encoder.int8.onnx"
	c.ModelConfig.Whisper.Decoder = "./sherpa-onnx-whisper-tiny.en/tiny.en-decoder.int8.onnx"
	c.ModelConfig.Tokens = "./sherpa-onnx-whisper-tiny.en/tiny.en-tokens.txt"
	c.ModelConfig.NumThreads = 2
	c.ModelConfig.Debug = 1
	c.ModelConfig.Provider = "cpu"

	recognizer := sherpa.NewOfflineRecognizer(&c)
	defer sherpa.DeleteOfflineRecognizer(recognizer)

	err := portaudio.Initialize()
	if err != nil {
		log.Fatalf("Unable to initialize portaudio: %v\n", err)
	}
	defer portaudio.Terminate()

	default_device, err := portaudio.DefaultInputDevice()
	if err != nil {
		log.Fatal("Failed to get default input device: %v\n", err)
	}
	log.Printf("Selected default input device: %s\n", default_device.Name)
	param := portaudio.StreamParameters{}
	param.Input.Device = default_device
	param.Input.Channels = 1
	param.Input.Latency = default_device.DefaultHighInputLatency

	param.SampleRate = float64(config.SampleRate)
	param.FramesPerBuffer = 0
	param.Flags = portaudio.ClipOff

	// you can choose another value for 0.1 if you want
	samplesPerCall := int32(param.SampleRate * 0.1) // 0.1 second
	samples := make([]float32, samplesPerCall)

	s, err := portaudio.OpenStream(param, samples)
	if err != nil {
		log.Fatalf("Failed to open the stream")
	}

	defer s.Close()
	chk(s.Start())

	log.Print("Started! Please speak")
	printed := false

	k := 0
	for {
		chk(s.Read())
		vad.AcceptWaveform(samples)

		if vad.IsSpeech() && !printed {
			printed = true
			log.Print("Detected speech\n")
		}

		if !vad.IsSpeech() {
			printed = false
		}

		for !vad.IsEmpty() {
			speechSegment := vad.Front()
			vad.Pop()

			duration := float32(len(speechSegment.Samples)) / float32(config.SampleRate)

			audio := &sherpa.Wave{}
			audio.Samples = speechSegment.Samples
			audio.SampleRate = config.SampleRate

			// Now decode it
			go decode(recognizer, audio, k)

			k += 1

			log.Printf("Duration: %.2f seconds\n", duration)
		}
	}

	chk(s.Stop())
}

func decode(recognizer *sherpa.OfflineRecognizer, audio *sherpa.Wave, id int) {
	stream := sherpa.NewOfflineStream(recognizer)
	defer sherpa.DeleteOfflineStream(stream)
	stream.AcceptWaveform(audio.SampleRate, audio.Samples)
	recognizer.Decode(stream)
	result := stream.GetResult()
	text := strings.ToLower(result.Text)
	text = strings.Trim(text, " ")
	log.Println(text)

	duration := float32(len(audio.Samples)) / float32(audio.SampleRate)

	filename := fmt.Sprintf("seg-%d-%.2f-seconds-%s.wav", id, duration, text)
	ok := audio.Save(filename)
	if ok {
		log.Printf("Saved to %s", filename)
	}
	log.Print("----------\n")
}

func chk(err error) {
	if err != nil {
		panic(err)
	}
}
