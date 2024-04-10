package main

import (
	"fmt"
	iso639 "github.com/barbashov/iso639-3"
	portaudio "github.com/csukuangfj/portaudio-go"
	sherpa "github.com/k2-fsa/sherpa-onnx-go/sherpa_onnx"
	"log"
)

func main() {
	log.SetFlags(log.LstdFlags | log.Lmicroseconds)

	// 1. Create VAD
	config := sherpa.VadModelConfig{}

	// Please download silero_vad.onnx from
	// https://github.com/snakers4/silero-vad/blob/master/files/silero_vad.onnx

	config.SileroVad.Model = "./silero_vad.onnx"
	config.SileroVad.Threshold = 0.5
	config.SileroVad.MinSilenceDuration = 0.5
	config.SileroVad.MinSpeechDuration = 0.25
	config.SileroVad.WindowSize = 512
	config.SampleRate = 16000
	config.NumThreads = 1
	config.Provider = "cpu"
	config.Debug = 1

	var bufferSizeInSeconds float32 = 20

	vad := sherpa.NewVoiceActivityDetector(&config, bufferSizeInSeconds)
	defer sherpa.DeleteVoiceActivityDetector(vad)

	// 2. Create spoken language identifier

	c := sherpa.SpokenLanguageIdentificationConfig{}
	c.Whisper.Encoder = "./sherpa-onnx-whisper-tiny/tiny-encoder.int8.onnx"
	c.Whisper.Decoder = "./sherpa-onnx-whisper-tiny/tiny-decoder.int8.onnx"
	c.NumThreads = 2
	c.Debug = 1
	c.Provider = "cpu"

	slid := sherpa.NewSpokenLanguageIdentification(&c)
	defer sherpa.DeleteSpokenLanguageIdentification(slid)

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
			go decode(slid, audio, k)

			k += 1

			log.Printf("Duration: %.2f seconds\n", duration)
		}
	}

	chk(s.Stop())
}

func decode(slid *sherpa.SpokenLanguageIdentification, audio *sherpa.Wave, id int) {
	stream := slid.CreateStream()
	defer sherpa.DeleteOfflineStream(stream)

	stream.AcceptWaveform(audio.SampleRate, audio.Samples)
	result := slid.Compute(stream)
	lang := iso639.FromPart1Code(result.Lang).Name
	log.Printf("Detected language: %v", lang)

	duration := float32(len(audio.Samples)) / float32(audio.SampleRate)

	filename := fmt.Sprintf("seg-%d-%.2f-seconds-%s.wav", id, duration, lang)
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
