package main

import (
	"encoding/json"
	"log"

	sherpa "github.com/k2-fsa/sherpa-onnx-go/sherpa_onnx"
	flag "github.com/spf13/pflag"
)

func main() {
	log.SetFlags(log.LstdFlags | log.Lmicroseconds)

	var referenceAudio string = "./sherpa-onnx-zipvoice-distill-int8-zh-en-emilia/test_wavs/leijun-1.wav"
	var referenceText string = "那还是三十六年前, 一九八七年. 我呢考上了武汉大学的计算机系."
	var outputFilename string = "./generated.wav"
	var numSteps int = 4
	var minCharInSentence int = 10

	text := "小米的价值观是真诚, 热爱. 真诚，就是不欺人也不自欺. 热爱, 就是全心投入并享受其中."

	flag.StringVar(&referenceAudio, "reference-audio", referenceAudio, "Path to the reference audio")
	flag.StringVar(&referenceText, "reference-text", referenceText, "Reference text for the reference audio")
	flag.StringVar(&text, "text", text, "Text to be synthesized")
	flag.StringVar(&outputFilename, "output-filename", outputFilename, "File to save the generated audio")
	flag.IntVar(&numSteps, "num-steps", numSteps, "Number of ZipVoice flow-matching steps")
	flag.IntVar(&minCharInSentence, "min-char-in-sentence", minCharInSentence, "Minimum characters in a sentence chunk")
	flag.Parse()

	var config sherpa.OfflineTtsConfig
	config.Model.Zipvoice.Encoder =
		"./sherpa-onnx-zipvoice-distill-int8-zh-en-emilia/encoder.int8.onnx"
	config.Model.Zipvoice.Decoder =
		"./sherpa-onnx-zipvoice-distill-int8-zh-en-emilia/decoder.int8.onnx"
	config.Model.Zipvoice.DataDir =
		"./sherpa-onnx-zipvoice-distill-int8-zh-en-emilia/espeak-ng-data"
	config.Model.Zipvoice.Lexicon =
		"./sherpa-onnx-zipvoice-distill-int8-zh-en-emilia/lexicon.txt"
	config.Model.Zipvoice.Tokens =
		"./sherpa-onnx-zipvoice-distill-int8-zh-en-emilia/tokens.txt"
	config.Model.Zipvoice.Vocoder = "./vocos_24khz.onnx"

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
	cfg.ReferenceText = referenceText
	cfg.NumSteps = numSteps

	extraMap := map[string]interface{}{
		"min_char_in_sentence": minCharInSentence,
	}
	extraBytes, err := json.Marshal(extraMap)
	if err != nil {
		log.Fatalf("Failed to marshal generation config extra: %v", err)
	}
	cfg.Extra = json.RawMessage(extraBytes)

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
