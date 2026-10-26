// Copyright (c)  2026  Xiaomi Corporation
//
// This file shows how to use the speech enhancement API with VoxLock models
// and a voice sample.
//
// VoxLock is a target-speaker voice isolation model. It keeps the voice of one
// speaker, e.g., the caller in a phone call, and removes the other voices.
//
// The model is from https://github.com/arbazsiddiqui/voxlock
//
// This example provides a voice sample of the speaker to keep, so VoxLock
// isolates that speaker from the first sample. The voice sample must have at
// least 3 seconds of the speaker's voice; from a longer file, VoxLock uses the
// loudest 3 seconds within its first 30 seconds. See
// ../speech-enhancement-voxlock if you don't have a voice sample.
//
// Please download the models from
// https://github.com/k2-fsa/sherpa-onnx/releases/tag/speech-enhancement-models
package main

import (
	sherpa "github.com/k2-fsa/sherpa-onnx-go/v2/sherpa_onnx"
	"log"
	"time"
)

func main() {
	log.SetFlags(log.LstdFlags | log.Lmicroseconds)

	config := sherpa.OfflineSpeechDenoiserConfig{}

	config.Model.VoxLock.EnrollModel = "./sherpa-onnx-voxlock-int8/enroll.onnx"
	config.Model.VoxLock.ChunkModel = "./sherpa-onnx-voxlock-int8/chunk_int8.onnx"
	config.Model.VoxLock.VoiceSample = "./sherpa-onnx-voxlock-int8/test_wavs/en-voice-sample-f.wav"
	config.Model.NumThreads = 1
	config.Model.Debug = 1

	sd := sherpa.NewOfflineSpeechDenoiser(&config)
	defer sherpa.DeleteOfflineSpeechDenoiser(sd)

	waveFilename := "./sherpa-onnx-voxlock-int8/test_wavs/en-combined.wav"

	wave := sherpa.ReadWave(waveFilename)
	if wave == nil {
		log.Printf("Failed to read %v\n", waveFilename)
		return
	}

	log.Println("Started")
	begin := time.Now()
	audio := sd.Run(wave.Samples, wave.SampleRate)
	elapsedSeconds := time.Since(begin).Seconds()
	log.Println("Done!")

	filename := "./voxlock-enhanced-with-voice-sample-16k.wav"
	ok := audio.Save(filename)
	if !ok {
		log.Fatalf("Failed to write %v\n", filename)
	} else {
		log.Println("Saved to ", filename)
	}

	duration := float64(len(wave.Samples)) / float64(wave.SampleRate)
	log.Printf("Elapsed seconds: %.3f\n", elapsedSeconds)
	log.Printf("Audio duration in seconds: %.3f\n", duration)
	log.Printf("RTF: %.3f/%.3f = %.3f\n", elapsedSeconds, duration, elapsedSeconds/duration)
}
