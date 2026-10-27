package main

import (
	"fmt"
	sherpa "github.com/k2-fsa/sherpa-onnx-go/sherpa_onnx"
	"log"
)

/*
Speaker diarization with Nemotron-3-Diarization, an end-to-end streaming
Sortformer model. It needs neither a speaker embedding model nor clustering.

Usage:

Step 1: Download the model

  wget https://github.com/k2-fsa/sherpa-onnx/releases/download/speaker-segmentation-models/sherpa-onnx-nemotron-3-diarization.tar.bz2
  tar xvf sherpa-onnx-nemotron-3-diarization.tar.bz2
  rm sherpa-onnx-nemotron-3-diarization.tar.bz2

Step 2. Download test wave files

  wget https://github.com/k2-fsa/sherpa-onnx/releases/download/speaker-segmentation-models/0-four-speakers-zh.wav

Step 3. Run it
*/

func initSpeakerDiarization() *sherpa.OfflineSpeakerDiarization {
	config := sherpa.OfflineSpeakerDiarizationConfig{}

	config.Segmentation.Sortformer.Model = "./sherpa-onnx-nemotron-3-diarization/model.int8.onnx"
	config.Segmentation.Sortformer.Threshold = 0.5
	config.Segmentation.NumThreads = 2

	config.MinDurationOn = 0.3
	config.MinDurationOff = 0.5

	sd := sherpa.NewOfflineSpeakerDiarization(&config)
	return sd
}

func main() {
	wave_filename := "./0-four-speakers-zh.wav"
	wave := sherpa.ReadWave(wave_filename)
	if wave == nil {
		log.Fatalf("Failed to read %v", wave_filename)
	}

	sd := initSpeakerDiarization()
	if sd == nil {
		log.Fatalf("Please check your config")
	}

	defer sherpa.DeleteOfflineSpeakerDiarization(sd)

	if wave.SampleRate != sd.SampleRate() {
		log.Fatalf("Expected sample rate: %v, given: %d", sd.SampleRate(), wave.SampleRate)
	}

	log.Println("Started")
	segments := sd.Process(wave.Samples)
	for _, s := range segments {
		fmt.Printf("%.3f -- %.3f speaker_%02d\n", s.Start, s.End, s.Speaker)
	}
}
