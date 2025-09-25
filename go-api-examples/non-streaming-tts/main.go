package main

import (
	"log"
	"math"

	sherpa "github.com/k2-fsa/sherpa-onnx-go/sherpa_onnx"
	flag "github.com/spf13/pflag"
)

func main() {
	log.SetFlags(log.LstdFlags | log.Lmicroseconds)

	config := sherpa.OfflineTtsConfig{}
	sid := 0
	filename := "./generated.wav"

	var zipvoiceNumSteps int
	var speed float32
	var promptText string
	var promptAudio string

	flag.StringVar(&config.Model.Vits.Model, "vits-model", "", "Path to the vits ONNX model")
	flag.StringVar(&config.Model.Vits.Lexicon, "vits-lexicon", "", "Path to lexicon.txt")
	flag.StringVar(&config.Model.Vits.Tokens, "vits-tokens", "", "Path to tokens.txt")
	flag.StringVar(&config.Model.Vits.DataDir, "vits-data-dir", "", "Path to espeak-ng-data")
	flag.StringVar(&config.Model.Vits.DictDir, "vits-dict-dir", "", "Path to dict for jieba")
	flag.Float32Var(&config.Model.Vits.NoiseScale, "vits-noise-scale", 0.667, "noise_scale for VITS")
	flag.Float32Var(&config.Model.Vits.NoiseScaleW, "vits-noise-scale-w", 0.8, "noise_scale_w for VITS")
	flag.Float32Var(&config.Model.Vits.LengthScale, "vits-length-scale", 1.0, "length_scale for VITS. small -> faster; large -> slower")

	flag.StringVar(&config.Model.Matcha.AcousticModel, "matcha-acoustic-model", "", "Path to the matcha acoustic model")
	flag.StringVar(&config.Model.Matcha.Vocoder, "matcha-vocoder", "", "Path to the matcha vocoder model")
	flag.StringVar(&config.Model.Matcha.Lexicon, "matcha-lexicon", "", "Path to lexicon.txt")
	flag.StringVar(&config.Model.Matcha.Tokens, "matcha-tokens", "", "Path to tokens.txt")
	flag.StringVar(&config.Model.Matcha.DataDir, "matcha-data-dir", "", "Path to espeak-ng-data")
	flag.StringVar(&config.Model.Matcha.DictDir, "matcha-dict-dir", "", "Path to dict for jieba")
	flag.Float32Var(&config.Model.Matcha.NoiseScale, "matcha-noise-scale", 0.667, "noise_scale for Matcha")
	flag.Float32Var(&config.Model.Matcha.LengthScale, "matcha-length-scale", 1.0, "length_scale for Matcha. small -> faster; large -> slower")

	flag.StringVar(&config.Model.Kokoro.Model, "kokoro-model", "", "Path to the Kokoro ONNX model")
	flag.StringVar(&config.Model.Kokoro.Voices, "kokoro-voices", "", "Path to voices.bin for Kokoro")
	flag.StringVar(&config.Model.Kokoro.Tokens, "kokoro-tokens", "", "Path to tokens.txt for Kokoro")
	flag.StringVar(&config.Model.Kokoro.DataDir, "kokoro-data-dir", "", "Path to espeak-ng-data for Kokoro")
	flag.StringVar(&config.Model.Kokoro.DictDir, "kokoro-dict-dir", "", "Path to dict for Kokoro")
	flag.StringVar(&config.Model.Kokoro.Lexicon, "kokoro-lexicon", "", "Path to lexicon files for Kokoro")
	flag.Float32Var(&config.Model.Kokoro.LengthScale, "kokoro-length-scale", 1.0, "length_scale for Kokoro. small -> faster; large -> slower")

	flag.StringVar(&config.Model.Kitten.Model, "kitten-model", "", "Path to the kitten ONNX model")
	flag.StringVar(&config.Model.Kitten.Voices, "kitten-voices", "", "Path to voices.bin for kitten")
	flag.StringVar(&config.Model.Kitten.Tokens, "kitten-tokens", "", "Path to tokens.txt for kitten")
	flag.StringVar(&config.Model.Kitten.DataDir, "kitten-data-dir", "", "Path to espeak-ng-data for kitten")
	flag.Float32Var(&config.Model.Kitten.LengthScale, "kitten-length-scale", 1.0, "length_scale for kitten. small -> faster; large -> slower")

	flag.StringVar(&config.Model.Zipvoice.Tokens, "zipvoice-tokens", "", "Path to tokens.txt for ZipVoice")
	flag.StringVar(&config.Model.Zipvoice.TextModel, "zipvoice-text-model", "", "Path to ZipVoice text encoder model")
	flag.StringVar(&config.Model.Zipvoice.FlowMatchingModel, "zipvoice-flow-matching-model", "", "Path to ZipVoice flow-matching decoder")
	flag.StringVar(&config.Model.Zipvoice.DataDir, "zipvoice-data-dir", "", "Path to espeak-ng-data")
	flag.StringVar(&config.Model.Zipvoice.PinyinDict, "zipvoice-pinyin-dict", "", "Path to pinyin.raw (for zh)")
	flag.StringVar(&config.Model.Zipvoice.Vocoder, "zipvoice-vocoder", "", "Path to vocoder (e.g., vocos_24khz.onnx)")

	flag.Float32Var(&config.Model.Zipvoice.FeatScale, "zipvoice-feat-scale", 0.1, "Feature scale for ZipVoice")
	flag.Float32Var(&config.Model.Zipvoice.TShift, "zipvoice-t-shift", 0.5, "t-shift for ZipVoice (smaller -> earlier t)")
	flag.Float32Var(&config.Model.Zipvoice.TargetRms, "zipvoice-target-rms", 0.1, "Target RMS for speech normalization (ZipVoice)")
	flag.Float32Var(&config.Model.Zipvoice.GuidanceScale, "zipvoice-guidance-scale", 1.0, "Classifier-free guidance scale (ZipVoice)")

	flag.IntVar(&zipvoiceNumSteps, "zipvoice-num-steps", 4, "Number of steps for ZipVoice inference")
	flag.Float32Var(&speed, "speed", 1.0, "Speech speed. larger->faster; smaller->slower")
	flag.StringVar(&promptText, "prompt-text", "", "Transcription of the prompt audio (ZipVoice)")
	flag.StringVar(&promptAudio, "prompt-audio", "", "Path to prompt audio wav (ZipVoice)")

	flag.IntVar(&config.Model.NumThreads, "num-threads", 1, "Number of threads for computing")
	flag.IntVar(&config.Model.Debug, "debug", 0, "Whether to show debug message")
	flag.StringVar(&config.Model.Provider, "provider", "cpu", "Provider to use: cpu/cuda/coreml")
	flag.StringVar(&config.RuleFsts, "tts-rule-fsts", "", "Path to rule.fst")
	flag.StringVar(&config.RuleFars, "tts-rule-fars", "", "Path to rule.far")
	flag.IntVar(&config.MaxNumSentences, "tts-max-num-sentences", 1, "Batch size (split long text to avoid OOM)")

	flag.IntVar(&sid, "sid", 0, "Speaker ID (multi-speaker models only)")
	flag.StringVar(&filename, "output-filename", "./generated.wav", "Output wav filename")

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
	var audio *sherpa.GeneratedAudio

	if promptAudio != "" {
		if promptText == "" {
			log.Fatal("For ZipVoice zero-shot TTS, --prompt-text is required when --prompt-audio is provided")
		}
		wave := sherpa.ReadWave(promptAudio)
		audio = tts.GenerateWithZipvoice(
			text,
			promptText,
			wave.Samples,
			wave.SampleRate,
			speed,
			zipvoiceNumSteps,
		)

	} else {
		audio = tts.Generate(text, sid, float32(math.Max(float64(speed), 1e-6)))
	}

	log.Println("Done!")
	if ok := audio.Save(filename); !ok {
		log.Fatalf("Failed to write %s", filename)
	}
	log.Println("Saved to", filename)
}
