package main

import (
	"log"
	"math"

	pp "github.com/csukuangfj/piper-phonemize-go/piper_phonemize"
	sherpa "github.com/k2-fsa/sherpa-onnx-go/sherpa_onnx"
	flag "github.com/spf13/pflag"
)

// This example shows the piper-phonemize text frontend for TTS.
//
// The text is phonemized outside of sherpa-onnx with
// https://github.com/csukuangfj/piper-phonemize-go and the resulting phoneme
// codepoints are passed to sherpa-onnx via GenerationConfig.PhonemeCodepoints.
// No lexicon file is needed. Please see
// https://github.com/k2-fsa/sherpa-onnx/tree/master/python-api-examples
// for the corresponding Python examples, e.g.,
// test-offline-tts-piper-phonemize.py
func main() {
	log.SetFlags(log.LstdFlags | log.Lmicroseconds)

	config := sherpa.OfflineTtsConfig{}
	lang := ""
	sid := 0
	filename := "./generated.wav"

	var speed float32

	flag.StringVar(&config.Model.Vits.Model, "vits-model", "", "Path to the vits ONNX model")
	flag.StringVar(&config.Model.Vits.Tokens, "vits-tokens", "", "Path to tokens.txt")
	flag.Float32Var(&config.Model.Vits.NoiseScale, "vits-noise-scale", 0.667, "noise_scale for VITS")
	flag.Float32Var(&config.Model.Vits.NoiseScaleW, "vits-noise-scale-w", 0.8, "noise_scale_w for VITS")
	flag.Float32Var(&config.Model.Vits.LengthScale, "vits-length-scale", 1.0, "length_scale for VITS. small -> faster; large -> slower")

	flag.StringVar(&config.Model.Matcha.AcousticModel, "matcha-acoustic-model", "", "Path to the matcha acoustic model")
	flag.StringVar(&config.Model.Matcha.Vocoder, "matcha-vocoder", "", "Path to the matcha vocoder model")
	flag.StringVar(&config.Model.Matcha.Tokens, "matcha-tokens", "", "Path to tokens.txt")
	flag.Float32Var(&config.Model.Matcha.NoiseScale, "matcha-noise-scale", 0.667, "noise_scale for Matcha")
	flag.Float32Var(&config.Model.Matcha.LengthScale, "matcha-length-scale", 1.0, "length_scale for Matcha. small -> faster; large -> slower")

	flag.StringVar(&config.Model.Kokoro.Model, "kokoro-model", "", "Path to the Kokoro ONNX model")
	flag.StringVar(&config.Model.Kokoro.Voices, "kokoro-voices", "", "Path to voices.bin for Kokoro")
	flag.StringVar(&config.Model.Kokoro.Tokens, "kokoro-tokens", "", "Path to tokens.txt for Kokoro")
	flag.Float32Var(&config.Model.Kokoro.LengthScale, "kokoro-length-scale", 1.0, "length_scale for Kokoro. small -> faster; large -> slower")

	flag.StringVar(&config.Model.Kitten.Model, "kitten-model", "", "Path to the kitten ONNX model")
	flag.StringVar(&config.Model.Kitten.Voices, "kitten-voices", "", "Path to voices.bin for kitten")
	flag.StringVar(&config.Model.Kitten.Tokens, "kitten-tokens", "", "Path to tokens.txt for kitten")
	flag.Float32Var(&config.Model.Kitten.LengthScale, "kitten-length-scale", 1.0, "length_scale for kitten. small -> faster; large -> slower")

	flag.StringVar(&lang, "lang", lang, "espeak-ng voice used by the piper phonemizer, e.g., en-us. Empty means use the language of the model")
	flag.Float32Var(&speed, "speed", 1.0, "Speech speed. larger->faster; smaller->slower")

	flag.IntVar(&config.Model.NumThreads, "num-threads", 1, "Number of threads for computing")
	flag.IntVar(&config.Model.Debug, "debug", 0, "Whether to show debug message")
	flag.StringVar(&config.Model.Provider, "provider", "cpu", "Provider to use: cpu/cuda/coreml")
	flag.StringVar(&config.RuleFsts, "tts-rule-fsts", "", "Path to rule.fst")
	flag.StringVar(&config.RuleFars, "tts-rule-fars", "", "Path to rule.far")
	flag.IntVar(&config.MaxNumSentences, "tts-max-num-sentences", 1, "Batch size (split long text to avoid OOM)")

	flag.IntVar(&sid, "sid", sid, "Speaker ID (multi-speaker models only)")
	flag.StringVar(&filename, "output-filename", filename, "Output wav filename")

	flag.Parse()

	if len(flag.Args()) != 1 {
		log.Fatalf("Please provide the text to generate audios")
	}
	text := flag.Arg(0)

	log.Println("Input text:", text)
	log.Println("Speaker ID:", sid)
	log.Println("Output filename:", filename)

	log.Println("Initializing model (may take several seconds)")
	tts := sherpa.NewOfflineTts(&config)
	defer sherpa.DeleteOfflineTts(tts)
	log.Println("Model created!")

	// Determine the language for the phonemizer. Use the one from the model
	// metadata unless --lang is given explicitly.
	if lang == "" {
		lang = tts.Lang()
		if lang == "" {
			log.Println("Warning: the model has no language information. " +
				"Falling back to en-us. Please use --lang to set it explicitly.")
			lang = "en-us"
		}
	}
	log.Println("Phonemizer language:", lang)

	// Phonemize the text outside of sherpa-onnx.
	//
	// Initialize("") uses the espeak-ng-data embedded in piper-phonemize-go.
	// It has nothing to do with sherpa-onnx; sherpa-onnx itself does not use
	// espeak-ng anymore.
	log.Println("piper-phonemize version:", pp.GetVersionStr())
	if ret := pp.Initialize(""); ret < 0 {
		log.Fatalf("Failed to initialize piper-phonemize")
	}

	result := pp.Phonemize(text, lang)
	if result == nil {
		log.Fatalf("Failed to phonemize the text")
	}
	defer pp.DeletePhonemizeResult(result)

	var sentences [][]int32
	for i := 0; i < result.GetNumSentences(); i++ {
		var s []int32
		for _, cp := range result.GetPhonemes(i) {
			s = append(s, int32(cp))
		}
		sentences = append(sentences, s)
	}
	log.Println("Number of phonemized sentences:", len(sentences))

	log.Println("Start generating!")
	cfg := sherpa.GenerationConfig{
		SilenceScale:      0.2,
		Speed:             float32(math.Max(float64(speed), 1e-6)),
		Sid:               sid,
		PhonemeCodepoints: sentences,
	}
	// The text has already been phonemized; pass an empty string here.
	audio := tts.GenerateWithConfig("", &cfg, nil)

	log.Println("Done!")
	if ok := audio.Save(filename); !ok {
		log.Fatalf("Failed to write %s", filename)
	}
	log.Println("Saved to", filename)
}
