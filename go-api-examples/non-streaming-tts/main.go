package main

import (
	sherpa "github.com/k2-fsa/sherpa-onnx-go/sherpa_onnx"
	flag "github.com/spf13/pflag"
	"log"
)

func main() {
	log.SetFlags(log.LstdFlags | log.Lmicroseconds)

	config := sherpa.OfflineTtsConfig{}
	sid := 0
	filename := "./generated.wav"

	flag.StringVar(&config.Model.Vits.Model, "vits-model", "", "Path to the vits ONNX model")
	flag.StringVar(&config.Model.Vits.Lexicon, "vits-lexicon", "", "Path to lexicon.txt")
	flag.StringVar(&config.Model.Vits.Tokens, "vits-tokens", "", "Path to tokens.txt")
	flag.StringVar(&config.Model.Vits.DataDir, "vits-data-dir", "", "Path to espeak-ng-data")

	flag.Float32Var(&config.Model.Vits.NoiseScale, "vits-noise-scale", 0.667, "noise_scale for VITS")
	flag.Float32Var(&config.Model.Vits.NoiseScaleW, "vits-noise-scale-w", 0.8, "noise_scale_w for VITS")
	flag.Float32Var(&config.Model.Vits.LengthScale, "vits-length-scale", 1.0, "length_scale for VITS. small -> faster in speech speed; large -> slower")

	flag.IntVar(&config.Model.NumThreads, "num-threads", 1, "Number of threads for computing")
	flag.IntVar(&config.Model.Debug, "debug", 0, "Whether to show debug message")
	flag.StringVar(&config.Model.Provider, "provider", "cpu", "Provider to use")
	flag.StringVar(&config.RuleFsts, "tts-rule-fsts", "", "Path to rule.fst")
	flag.StringVar(&config.RuleFars, "tts-rule-fars", "", "Path to rule.far")
	flag.IntVar(&config.MaxNumSentences, "tts-max-num-sentences", 1, "Batch size")

	flag.IntVar(&sid, "sid", 0, "Speaker ID. Used only for multi-speaker models")
	flag.StringVar(&filename, "output-filename", "./generated.wav", "Filename to save the generated audio")

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

	log.Println("Start generating!")

	audio := tts.Generate(text, sid, 1.0)

	log.Println("Done!")

	ok := audio.Save(filename)
	if !ok {
		log.Fatalf("Failed to write", filename)
	}
}
