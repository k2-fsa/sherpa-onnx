package main

import (
	"encoding/json"
	"log"

	sherpa "github.com/k2-fsa/sherpa-onnx-go/sherpa_onnx"
)

func main() {
	log.SetFlags(log.LstdFlags | log.Lmicroseconds)

	// ---------------- config ----------------
	var config sherpa.OfflineTtsConfig

	config.Model.Supertonic.DurationPredictor =
		"./sherpa-onnx-supertonic-tts-int8-2026-03-06/duration_predictor.int8.onnx"
	config.Model.Supertonic.TextEncoder =
		"./sherpa-onnx-supertonic-tts-int8-2026-03-06/text_encoder.int8.onnx"
	config.Model.Supertonic.VectorEstimator =
		"./sherpa-onnx-supertonic-tts-int8-2026-03-06/vector_estimator.int8.onnx"
	config.Model.Supertonic.Vocoder =
		"./sherpa-onnx-supertonic-tts-int8-2026-03-06/vocoder.int8.onnx"
	config.Model.Supertonic.TtsJson =
		"./sherpa-onnx-supertonic-tts-int8-2026-03-06/tts.json"
	config.Model.Supertonic.UnicodeIndexer =
		"./sherpa-onnx-supertonic-tts-int8-2026-03-06/unicode_indexer.bin"
	config.Model.Supertonic.VoiceStyle =
		"./sherpa-onnx-supertonic-tts-int8-2026-03-06/voice.bin"

	config.Model.NumThreads = 2
	config.Model.Debug = 1

	log.Println("Creating Offline TTS")
	tts := sherpa.NewOfflineTts(&config)
	if tts == nil {
		log.Fatal("Failed to create OfflineTts")
	}
	defer sherpa.DeleteOfflineTts(tts)

	text := "Today as always, men fall into two groups: slaves and free men. Whoever " +
		"does not have two-thirds of his day for himself, is a slave, whatever " +
		"he may be: a statesman, a businessman, an official, or a scholar."

	var cfg sherpa.GenerationConfig
	cfg.Sid = 6
	cfg.NumSteps = 5
	cfg.Speed = 1.25 // larger -> faster

	extraMap := map[string]interface{}{
		"lang": "en",
	}
	extraBytes, _ := json.Marshal(extraMap)
	cfg.Extra = json.RawMessage(extraBytes)

	log.Println("Start generating")

	audio := tts.GenerateWithConfig(
		text,
		&cfg,
		func(samples []float32, progress float32) bool {
			log.Printf("Progress: %.3f%%, Number of samples: %d", progress*100, len(samples))
			return true
		},
	)

	if audio == nil {
		log.Fatal("Generation failed")
	}

	outputFilename := "./generated-supertonic-en.wav"
	if !audio.Save(outputFilename) {
		log.Fatal("Failed to save wav")
	}

	log.Println("Saved to:", outputFilename)
}
