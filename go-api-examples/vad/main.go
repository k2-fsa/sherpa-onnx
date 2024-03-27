package main

import (
	"fmt"
	"github.com/gordonklaus/portaudio"
	sherpa "github.com/k2-fsa/sherpa-onnx-go/sherpa_onnx"
	"log"
)

func main() {
	log.SetFlags(log.LstdFlags | log.Lmicroseconds)

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

	var bufferSizeInSeconds float32 = 5

	vad := sherpa.NewVoiceActivityDetector(&config, bufferSizeInSeconds)
	defer sherpa.DeleteVoiceActivityDetector(vad)

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
	param.Input.Latency = default_device.DefaultLowInputLatency

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

			audio := sherpa.GeneratedAudio{}
			audio.Samples = speechSegment.Samples
			audio.SampleRate = config.SampleRate

			filename := fmt.Sprintf("seg-%d-%.2f-seconds.wav", k, duration)
			ok := audio.Save(filename)
			if ok {
				log.Printf("Saved to %s", filename)
			}

			k += 1

			log.Printf("Duration: %.2f seconds\n", duration)
			log.Print("----------\n")
		}
	}

	chk(s.Stop())
}

func chk(err error) {
	if err != nil {
		panic(err)
	}
}
