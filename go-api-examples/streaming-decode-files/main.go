package main

import (
	"log"
	"strings"

	sherpa "github.com/k2-fsa/sherpa-onnx-go/sherpa_onnx"
	flag "github.com/spf13/pflag"
)

func main() {
	log.Printf("sherpa-onnx version: %v\n", sherpa.GetVersion())
	log.Printf("sherpa-onnx gitSha1: %v\n", sherpa.GetGitSha1())
	log.Printf("sherpa-onnx gitDate: %v\n", sherpa.GetGitDate())

	log.SetFlags(log.LstdFlags | log.Lmicroseconds)

	config := sherpa.OnlineRecognizerConfig{}
	config.FeatConfig = sherpa.FeatureConfig{SampleRate: 16000, FeatureDim: 80}

	flag.StringVar(&config.ModelConfig.Transducer.Encoder, "encoder", "", "Path to the transducer encoder model")
	flag.StringVar(&config.ModelConfig.Transducer.Decoder, "decoder", "", "Path to the transducer decoder model")
	flag.StringVar(&config.ModelConfig.Transducer.Joiner, "joiner", "", "Path to the transducer joiner model")
	flag.StringVar(&config.ModelConfig.Paraformer.Encoder, "paraformer-encoder", "", "Path to the paraformer encoder model")
	flag.StringVar(&config.ModelConfig.Paraformer.Decoder, "paraformer-decoder", "", "Path to the paraformer decoder model")
	flag.StringVar(&config.ModelConfig.Zipformer2Ctc.Model, "zipformer2-ctc", "", "Path to the zipformer2 CTC model")
	flag.StringVar(&config.ModelConfig.ToneCtc.Model, "t-one-ctc", "", "Path to the T-one CTC model")
	flag.StringVar(&config.ModelConfig.Tokens, "tokens", "", "Path to the tokens file")
	flag.IntVar(&config.ModelConfig.NumThreads, "num-threads", 1, "Number of threads for computing")
	flag.IntVar(&config.ModelConfig.Debug, "debug", 0, "Whether to show debug message")
	flag.StringVar(&config.ModelConfig.ModelType, "model-type", "", "Optional. Used for loading the model in a faster way")
	flag.StringVar(&config.ModelConfig.Provider, "provider", "cpu", "Provider to use")
	flag.StringVar(&config.DecodingMethod, "decoding-method", "greedy_search", "Decoding method. Possible values: greedy_search, modified_beam_search")
	flag.IntVar(&config.MaxActivePaths, "max-active-paths", 4, "Used only when --decoding-method is modified_beam_search")
	flag.StringVar(&config.RuleFsts, "rule-fsts", "", "If not empty, path to rule fst for inverse text normalization")
	flag.StringVar(&config.RuleFars, "rule-fars", "", "If not empty, path to rule fst archives for inverse text normalization")
	flag.StringVar(&config.Hr.Lexicon, "hr-lexicon", "", "If not empty, path to the lexicon.txt for homonphone replacer")
	flag.StringVar(&config.Hr.RuleFsts, "hr-rule-fsts", "", "If not empty, path to the replace.fst for homonphone replacer")

	flag.Parse()

	if len(flag.Args()) != 1 {
		log.Fatalf("Please provide one wave file")
	}

	waveFilename := flag.Arg(0)
	log.Println("Reading", waveFilename)

	wave := sherpa.ReadWave(waveFilename)
	if wave == nil {
		log.Fatalf("Failed to read %v", waveFilename)
	}

	log.Println("Initializing recognizer (may take several seconds)")
	recognizer := sherpa.NewOnlineRecognizer(&config)
	log.Println("Recognizer created!")
	defer sherpa.DeleteOnlineRecognizer(recognizer)

	log.Println("Start decoding!")
	stream := sherpa.NewOnlineStream(recognizer)
	defer sherpa.DeleteOnlineStream(stream)

	leftPadding := make([]float32, int(float32(wave.SampleRate)*0.3))
	stream.AcceptWaveform(wave.SampleRate, leftPadding)

	stream.AcceptWaveform(wave.SampleRate, wave.Samples)

	tailPadding := make([]float32, int(float32(wave.SampleRate)*0.6))
	stream.AcceptWaveform(wave.SampleRate, tailPadding)

	for recognizer.IsReady(stream) {
		recognizer.Decode(stream)
	}
	log.Println("Decoding done!")
	result := recognizer.GetResult(stream)
	log.Println(strings.ToLower(result.Text))
	log.Printf("Wave duration: %v seconds", float32(len(wave.Samples))/float32(wave.SampleRate))
}
