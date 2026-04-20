package main

import (
	sherpa "github.com/k2-fsa/sherpa-onnx-go/sherpa_onnx"
	"log"
)

func appendSamples(dst []float32, src []float32) []float32 {
	return append(dst, src...)
}

func main() {
	log.SetFlags(log.LstdFlags | log.Lmicroseconds)

	config := sherpa.OnlineSpeechDenoiserConfig{}
	config.Model.Gtcrn.Model = "./gtcrn_simple.onnx"
	config.Model.NumThreads = 1
	config.Model.Debug = 1

	sd := sherpa.NewOnlineSpeechDenoiser(&config)
	defer sherpa.DeleteOnlineSpeechDenoiser(sd)

	waveFilename := "./inp_16k.wav"
	wave := sherpa.ReadWave(waveFilename)
	if wave == nil {
		log.Printf("Failed to read %v\n", waveFilename)
		return
	}

	output := make([]float32, 0, len(wave.Samples))
	frameShift := sd.FrameShiftInSamples()
	for start := 0; start < len(wave.Samples); start += frameShift {
		end := start + frameShift
		if end > len(wave.Samples) {
			end = len(wave.Samples)
		}
		audio := sd.Run(wave.Samples[start:end], wave.SampleRate)
		output = appendSamples(output, audio.Samples)
	}

	output = appendSamples(output, sd.Flush().Samples)
	filename := "./enhanced-online-gtcrn.wav"
	if !(&sherpa.DenoisedAudio{Samples: output, SampleRate: sd.SampleRate()}).Save(filename) {
		log.Fatalf("Failed to write %v\n", filename)
	}

	log.Printf("Saved to %v\n", filename)
}
