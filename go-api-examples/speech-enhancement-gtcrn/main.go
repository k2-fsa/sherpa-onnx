package main

import (
	sherpa "github.com/k2-fsa/sherpa-onnx-go/sherpa_onnx"
	"log"
)

func main() {
	log.SetFlags(log.LstdFlags | log.Lmicroseconds)

	config := sherpa.OfflineSpeechDenoiserConfig{}

	// Please download the models from
	// https://github.com/k2-fsa/sherpa-onnx/releases/tag/speech-enhancement-models

	config.Model.Gtcrn.Model = "./gtcrn_simple.onnx"
	config.Model.NumThreads = 1
	config.Model.Debug = 1

	sd := sherpa.NewOfflineSpeechDenoiser(&config)
	defer sherpa.DeleteOfflineSpeechDenoiser(sd)

	wave_filename := "./inp_16k.wav"

	wave := sherpa.ReadWave(wave_filename)
	if wave == nil {
		log.Printf("Failed to read %v\n", wave_filename)
		return
	}

	log.Println("Started")
	audio := sd.Run(wave.Samples, wave.SampleRate)
	log.Println("Done!")

	filename := "./enhanced-16k.wav"
	ok := audio.Save(filename)
	if !ok {
		log.Fatalf("Failed to write", filename)
	} else {
		log.Println("Saved to ", filename)
	}

}
