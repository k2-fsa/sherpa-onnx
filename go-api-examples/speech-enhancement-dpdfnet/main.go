package main

import (
	sherpa "github.com/k2-fsa/sherpa-onnx-go/sherpa_onnx"
	"log"
)

func main() {
	log.SetFlags(log.LstdFlags | log.Lmicroseconds)

	config := sherpa.OfflineSpeechDenoiserConfig{}
	config.Model.DpdfNet.Model = "./dpdfnet_baseline.onnx"
	config.Model.NumThreads = 1
	config.Model.Debug = 1

	sd := sherpa.NewOfflineSpeechDenoiser(&config)
	defer sherpa.DeleteOfflineSpeechDenoiser(sd)

	waveFilename := "./inp_16k.wav"
	wave := sherpa.ReadWave(waveFilename)
	if wave == nil {
		log.Printf("Failed to read %v\n", waveFilename)
		return
	}

	audio := sd.Run(wave.Samples, wave.SampleRate)
	filename := "./enhanced-dpdfnet-16k.wav"
	if !audio.Save(filename) {
		log.Fatalf("Failed to write %v\n", filename)
	}

	log.Printf("Saved to %v\n", filename)
}
