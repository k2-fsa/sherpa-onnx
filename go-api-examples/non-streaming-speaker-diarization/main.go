package main

import (
	"fmt"
	sherpa "github.com/k2-fsa/sherpa-onnx-go/sherpa_onnx"
	"log"
)

/*
Usage:

Step 1: Download a speaker segmentation model

Please visit https://github.com/k2-fsa/sherpa-onnx/releases/tag/speaker-segmentation-models
for a list of available models. The following is an example

  wget https://github.com/k2-fsa/sherpa-onnx/releases/download/speaker-segmentation-models/sherpa-onnx-pyannote-segmentation-3-0.tar.bz2
  tar xvf sherpa-onnx-pyannote-segmentation-3-0.tar.bz2
  rm sherpa-onnx-pyannote-segmentation-3-0.tar.bz2

Step 2: Download a speaker embedding extractor model

Please visit https://github.com/k2-fsa/sherpa-onnx/releases/tag/speaker-recongition-models
for a list of available models. The following is an example

  wget https://github.com/k2-fsa/sherpa-onnx/releases/download/speaker-recongition-models/3dspeaker_speech_eres2net_base_sv_zh-cn_3dspeaker_16k.onnx

Step 3. Download test wave files

Please visit https://github.com/k2-fsa/sherpa-onnx/releases/tag/speaker-segmentation-models
for a list of available test wave files. The following is an example

  wget https://github.com/k2-fsa/sherpa-onnx/releases/download/speaker-segmentation-models/0-four-speakers-zh.wav

Step 4. Run it
*/

func initSpeakerDiarization() *sherpa.OfflineSpeakerDiarization {
	config := sherpa.OfflineSpeakerDiarizationConfig{}

	config.Segmentation.Pyannote.Model = "./sherpa-onnx-pyannote-segmentation-3-0/model.onnx"
	config.Segmentation.Pyannote.WindowShiftRatio = 0.1
	config.Embedding.Model = "./3dspeaker_speech_eres2net_base_sv_zh-cn_3dspeaker_16k.onnx"

	// The test wave file contains 4 speakers, so we use 4 here
	config.Clustering.NumClusters = 4

	// if you don't know the actual numbers in the wave file,
	// then please don't set NumClusters; you need to use
	//
	// config.Clustering.Threshold = 0.5
	//

	// A larger Threshold leads to fewer clusters
	// A smaller Threshold leads to more clusters

	// Set ComputeConfidence to 1 to enable per-segment confidence computation
	// config.Clustering.ComputeConfidence = 1

	sd := sherpa.NewOfflineSpeakerDiarization(&config)
	return sd
}

func main() {
	wave_filename := "./0-four-speakers-zh.wav"
	wave := sherpa.ReadWave(wave_filename)
	if wave == nil {
		log.Printf("Failed to read %v", wave_filename)
		return
	}

	sd := initSpeakerDiarization()
	if sd == nil {
		log.Printf("Please check your config")
		return
	}

	defer sherpa.DeleteOfflineSpeakerDiarization(sd)

	if wave.SampleRate != sd.SampleRate() {
		log.Printf("Expected sample rate: %v, given: %d\n", sd.SampleRate(), wave.SampleRate)
		return
	}

	log.Println("Started")
	segments := sd.Process(wave.Samples)
	n := len(segments)

	const confidenceUnavailable = -2.0
	for i := 0; i < n; i++ {
		suffix := ""
		if float64(segments[i].Confidence) != confidenceUnavailable {
			suffix = fmt.Sprintf(" confidence=%.3f", segments[i].Confidence)
		}
		log.Printf("%.3f -- %.3f speaker_%02d%s\n",
			segments[i].Start, segments[i].End, segments[i].Speaker, suffix)
	}
}
