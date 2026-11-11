// Copyright (c)  2026  Xiaomi Corporation
//
// This file shows how to use the online speech enhancement API with VoxLock
// models.
//
// VoxLock is a target-speaker voice isolation model. It keeps the voice of one
// speaker, e.g., the caller in a phone call, and removes the other voices.
//
// The model is from https://github.com/arbazsiddiqui/voxlock
//
// This example does not provide a voice sample, so VoxLock passes the audio
// through unchanged until it has heard 3 seconds of speech, takes that speech
// as the speaker to keep, and isolates from then on. See
// ../streaming-speech-enhancement-voxlock-with-voice-sample if you have a
// voice sample of the speaker.
//
// Please download the models from
// https://github.com/k2-fsa/sherpa-onnx/releases/tag/speech-enhancement-models
package main

import (
	sherpa "github.com/k2-fsa/sherpa-onnx-go/v2/sherpa_onnx"
	"log"
	"time"
)

func appendSamples(dst []float32, src []float32) []float32 {
	return append(dst, src...)
}

func main() {
	log.SetFlags(log.LstdFlags | log.Lmicroseconds)

	config := sherpa.OnlineSpeechDenoiserConfig{}
	config.Model.VoxLock.EnrollModel = "./sherpa-onnx-voxlock-int8/enroll.onnx"
	config.Model.VoxLock.ChunkModel = "./sherpa-onnx-voxlock-int8/chunk_int8.onnx"
	config.Model.NumThreads = 1
	config.Model.Debug = 1

	sd := sherpa.NewOnlineSpeechDenoiser(&config)
	defer sherpa.DeleteOnlineSpeechDenoiser(sd)

	waveFilename := "./sherpa-onnx-voxlock-int8/test_wavs/en-combined.wav"
	wave := sherpa.ReadWave(waveFilename)
	if wave == nil {
		log.Printf("Failed to read %v\n", waveFilename)
		return
	}

	output := make([]float32, 0, len(wave.Samples))
	frameShift := sd.FrameShiftInSamples()

	log.Println("Started")
	begin := time.Now()
	for start := 0; start < len(wave.Samples); start += frameShift {
		end := start + frameShift
		if end > len(wave.Samples) {
			end = len(wave.Samples)
		}
		audio := sd.Run(wave.Samples[start:end], wave.SampleRate)
		output = appendSamples(output, audio.Samples)
	}

	output = appendSamples(output, sd.Flush().Samples)
	elapsedSeconds := time.Since(begin).Seconds()
	log.Println("Done!")

	filename := "./voxlock-enhanced-online-16k.wav"
	if !(&sherpa.DenoisedAudio{Samples: output, SampleRate: sd.SampleRate()}).Save(filename) {
		log.Fatalf("Failed to write %v\n", filename)
	}

	log.Printf("Saved to %v\n", filename)

	duration := float64(len(wave.Samples)) / float64(wave.SampleRate)
	log.Printf("Elapsed seconds: %.3f\n", elapsedSeconds)
	log.Printf("Audio duration in seconds: %.3f\n", duration)
	log.Printf("RTF: %.3f/%.3f = %.3f\n", elapsedSeconds, duration, elapsedSeconds/duration)
}
