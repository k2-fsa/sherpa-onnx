package main

import (
	"github.com/gordonklaus/portaudio"
	sherpa "github.com/k2-fsa/sherpa-onnx-go/sherpa_onnx"
	"log"
)

func testCircularBuffer() {
	buffer := sherpa.NewCircularBuffer(10)
	defer sherpa.DeleteCircularBuffer(buffer)

	samples := []float32{1, 2, 3}
	buffer.Push(samples)
	log.Println("head %v, size: %v!", buffer.Head(), buffer.Size())
	buffer.Pop(1)
	log.Println("head %v, size: %v!", buffer.Head(), buffer.Size())

	buffer.Push([]float32{10, 20, 30, 40})
	log.Println("head %v, size: %v!", buffer.Head(), buffer.Size())

	p := buffer.Get(1, 5)
	log.Println("head %v, size: %v, %v", buffer.Head(), buffer.Size(), p)
}

func main() {
	log.SetFlags(log.LstdFlags | log.Lmicroseconds)

	testCircularBuffer()

	config := sherpa.VadModelConfig{}
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

	vad.AcceptWaveform([]float32{0.2, 0.3, 0.4, 0.5})

	log.Println("%v", vad.IsEmpty())

}

func chk(err error) {
	if err != nil {
		panic(err)
	}
}
