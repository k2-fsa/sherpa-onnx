package main

import (
	"log"

	"encoding/json"
	sherpa "github.com/k2-fsa/sherpa-onnx-go/sherpa_onnx"
	flag "github.com/spf13/pflag"
)

func main() {
	log.SetFlags(log.LstdFlags | log.Lmicroseconds)

	var referenceAudio string = "./sherpa-onnx-pocket-tts-int8-2026-01-26/test_wavs/bria.wav"
	var outputFilename string = "./generated.wav"

	text := `Today as always, men fall into two groups: slaves and free men.
Whoever does not have two-thirds of his day for himself, is a slave,
whatever he may be: a statesman, a businessman, an official, or a scholar.`

	flag.StringVar(&referenceAudio, "reference-audio", "", "Path to the reference audio")
	flag.StringVar(&text, "text", text, "Text to be synthesized")
	flag.StringVar(&outputFilename, "output-filename", outputFilename, "File to save the generated audio")
	flag.Parse()

	// ---------------- config ----------------
	var config sherpa.OfflineTtsConfig

	config.Model.Pocket.LmFlow =
		"./sherpa-onnx-pocket-tts-int8-2026-01-26/lm_flow.int8.onnx"
	config.Model.Pocket.LmMain =
		"./sherpa-onnx-pocket-tts-int8-2026-01-26/lm_main.int8.onnx"
	config.Model.Pocket.Encoder =
		"./sherpa-onnx-pocket-tts-int8-2026-01-26/encoder.onnx"
	config.Model.Pocket.Decoder =
		"./sherpa-onnx-pocket-tts-int8-2026-01-26/decoder.int8.onnx"
	config.Model.Pocket.TextConditioner =
		"./sherpa-onnx-pocket-tts-int8-2026-01-26/text_conditioner.onnx"
	config.Model.Pocket.VocabJson =
		"./sherpa-onnx-pocket-tts-int8-2026-01-26/vocab.json"
	config.Model.Pocket.TokenScoresJson =
		"./sherpa-onnx-pocket-tts-int8-2026-01-26/token_scores.json"

	config.Model.NumThreads = 2
	config.Model.Debug = 0
	config.Model.Provider = "cpu"

	log.Println("Creating Offline TTS")
	tts := sherpa.NewOfflineTts(&config)
	if tts == nil {
		log.Fatal("Failed to create OfflineTts")
	}
	defer sherpa.DeleteOfflineTts(tts)

	wave := sherpa.ReadWave(referenceAudio)
	if wave == nil {
		log.Fatal("Failed to read reference wav:", referenceAudio)
	}

	var cfg sherpa.GenerationConfig
	cfg.ReferenceAudio = wave.Samples
	cfg.ReferenceSampleRate = wave.SampleRate

	cfg.Extra = json.RawMessage(`{
  "max_reference_audio_len": 10,
	"temperature": 0.7,
	}`)

	log.Println("Start generating")

	audio := tts.GenerateWithConfig(
		text,
		&cfg,
		func(samples []float32, progress float32) bool {
			log.Printf("Progress: %.3f%%, Number of samples: %d", progress*100, len(samples))
			// return false here if you want to cancel
			return true
		},
	)

	if audio == nil {
		log.Fatal("Generation failed")
	}

	if !audio.Save(outputFilename) {
		log.Fatal("Failed to save wav")
	}

	log.Println("Saved to:", outputFilename)
}
