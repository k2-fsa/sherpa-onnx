package main

import (
	"fmt"
	"github.com/gen2brain/malgo"
	sherpa "github.com/k2-fsa/sherpa-onnx-go/sherpa_onnx"
	"log"
)

func main() {
	log.SetFlags(log.LstdFlags | log.Lmicroseconds)

	config := sherpa.VadModelConfig{}

	// Please download silero_vad.onnx from
	// https://github.com/snakers4/silero-vad/raw/master/src/silero_vad/data/silero_vad.onnx

	config.SileroVad.Model = "./silero_vad.onnx"
	config.SileroVad.Threshold = 0.5
	config.SileroVad.MinSilenceDuration = 0.5
	config.SileroVad.MinSpeechDuration = 0.25
	config.SileroVad.WindowSize = 512
	config.SampleRate = 16000
	config.NumThreads = 1
	config.Provider = "cpu"
	config.Debug = 1

	window_size := config.SileroVad.WindowSize

	var bufferSizeInSeconds float32 = 5

	vad := sherpa.NewVoiceActivityDetector(&config, bufferSizeInSeconds)
	defer sherpa.DeleteVoiceActivityDetector(vad)

	buffer := sherpa.NewCircularBuffer(10 * config.SampleRate)
	defer sherpa.DeleteCircularBuffer(buffer)

	ctx, err := malgo.InitContext(nil, malgo.ContextConfig{}, func(message string) {
		fmt.Printf("LOG <%v>", message)
	})
	chk(err)

	defer func() {
		_ = ctx.Uninit()
		ctx.Free()
	}()

	deviceConfig := malgo.DefaultDeviceConfig(malgo.Duplex)
	deviceConfig.Capture.Format = malgo.FormatS16
	deviceConfig.Capture.Channels = 1
	deviceConfig.Playback.Format = malgo.FormatS16
	deviceConfig.Playback.Channels = 1
	deviceConfig.SampleRate = 16000
	deviceConfig.Alsa.NoMMap = 1

	printed := false
	k := 0

	onRecvFrames := func(_, pSample []byte, framecount uint32) {
		samples := samplesInt16ToFloat(pSample)
		buffer.Push(samples)
		for buffer.Size() >= window_size {
			head := buffer.Head()
			s := buffer.Get(head, window_size)
			buffer.Pop(window_size)

			vad.AcceptWaveform(s)

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
	}

	captureCallbacks := malgo.DeviceCallbacks{
		Data: onRecvFrames,
	}

	device, err := malgo.InitDevice(ctx.Context, deviceConfig, captureCallbacks)
	chk(err)

	err = device.Start()
	chk(err)

	fmt.Println("Started. Please speak. Press ctrl + C  to exit")
	fmt.Scanln()
	device.Uninit()

}

func chk(err error) {
	if err != nil {
		panic(err)
	}
}

func samplesInt16ToFloat(inSamples []byte) []float32 {
	numSamples := len(inSamples) / 2
	outSamples := make([]float32, numSamples)

	for i := 0; i != numSamples; i++ {
		// Decode two bytes into an int16 using bit manipulation
		s16 := int16(inSamples[2*i]) | int16(inSamples[2*i+1])<<8
		outSamples[i] = float32(s16) / 32768
	}

	return outSamples
}
