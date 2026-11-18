package main

import (
	"log"
	"os"
	"time"

	sherpa "github.com/k2-fsa/sherpa-onnx-go/v2/sherpa_onnx"
)

func main() {
	modelDir := "./english-mfa-v3.1.0"
	wav := "./english-mfa-v3.1.0/en.wav"
	// The transcript must match the wav.
	text := "ask not what your country can do for you, ask what you can do for your country."

	config := &sherpa.ForcedAlignerConfig{}
	config.ModelDir = modelDir
	config.Lexicon = modelDir + "/lexicon.txt"

	aligner := sherpa.NewForcedAligner(config)
	if aligner == nil {
		log.Fatalf("Failed to create the forced aligner")
	}
	defer sherpa.DeleteForcedAligner(aligner)

	wave := sherpa.ReadWave(wav)

	stream := aligner.CreateStream()
	defer sherpa.DeleteOfflineStream(stream)
	stream.AcceptWaveform(wave.SampleRate, wave.Samples)

	start := time.Now()
	result := aligner.Compute(stream, text)
	elapsed := time.Since(start).Seconds()
	if result == nil {
		log.Fatalf("Failed to align")
	}

	// tiers[0] is the "phones" tier; every interval has a phone symbol.
	log.Printf("phones:")
	for _, iv := range result.Tiers[0].Intervals {
		log.Printf("%.3f - %.3f  %s", iv.Start, iv.End, iv.Text)
	}

	// tiers[1] is the "words" tier; an empty text marks a silence gap.
	log.Printf("words:")
	for _, iv := range result.Tiers[1].Intervals {
		label := iv.Text
		if label == "" {
			label = "<silence>"
		}
		log.Printf("%.3f - %.3f  %s", iv.Start, iv.End, label)
	}

	output := "en.TextGrid"
	if err := os.WriteFile(output, []byte(result.TextGrid), 0644); err != nil {
		log.Fatalf("Failed to write %v: %v", output, err)
	}
	log.Printf("Saved to %v", output)

	duration := float32(len(wave.Samples)) / float32(wave.SampleRate)
	rtf := elapsed / float64(duration)
	log.Printf("Elapsed seconds: %.3f", elapsed)
	log.Printf("Audio duration in seconds: %.3f", duration)
	log.Printf("RTF: %.3f/%.3f = %.3f", elapsed, duration, rtf)
}
