package main

import (
	"bytes"
	"encoding/binary"
	"github.com/DylanMeeus/GoAudio/wave"
	sherpa "github.com/k2-fsa/sherpa-onnx-go/sherpa_onnx"
	flag "github.com/spf13/pflag"

	"log"
)

func main() {
	log.SetFlags(log.LstdFlags | log.Lmicroseconds)

	config := sherpa.OnlineRecognizerConfig{}
	config.FeatConfig = sherpa.FeatureConfig{SampleRate: 16000, FeatureDim: 80}

	flag.StringVar(&config.ModelConfig.Encoder, "encoder", "", "Path to the encoder model")
	flag.StringVar(&config.ModelConfig.Decoder, "decoder", "", "Path to the decoder model")
	flag.StringVar(&config.ModelConfig.Joiner, "joiner", "", "Path to the joiner model")
	flag.StringVar(&config.ModelConfig.Tokens, "tokens", "", "Path to the tokens file")
	flag.IntVar(&config.ModelConfig.NumThreads, "num-threads", 1, "Number of threads for computing")
	flag.IntVar(&config.ModelConfig.Debug, "debug", 0, "Whether to show debug message")
	flag.StringVar(&config.ModelConfig.ModelType, "model-type", "", "Optional. Used for loading the model in a faster way")
	flag.StringVar(&config.ModelConfig.Provider, "provider", "cpu", "Provider to use")
	flag.StringVar(&config.DecodingMethod, "decoding-method", "greedy_search", "Decoding method. Possible values: greedy_search, modified_beam_search")
	flag.IntVar(&config.MaxActivePaths, "max-active-paths", 4, "Used only when --decoding-method is modified_beam_search")

	flag.Parse()

	if len(flag.Args()) != 1 {
		log.Fatalf("Please provide one wave file")
	}

	log.Println("Reading", flag.Arg(0))
	wav, err := wave.ReadWaveFile(flag.Arg(0))
	if err != nil {
		log.Fatalf("Failed to read wave file: %s, %v", flag.Arg(0), err)
	}
	log.Printf("Sample rate: %v", wav.SampleRate)
	log.Printf("Bits per sample: %v", wav.BitsPerSample)

	if wav.NumChannels != 1 {
		log.Fatalf("Only single channel is supported. Given: %v", wav.NumChannels)
	}

	if wav.BitsPerSample != 16 {
		log.Fatalf("Only 16-bit per sample is supported. Given: %v", wav.BitsPerSample)
	}

	samplesf32 := samplesInt16ToFloat(wav.RawData)
	log.Printf("Duration: %v seconds", float32(len(samplesf32))/float32(wav.SampleRate))

	log.Println("Initializing recognizer (may take several seconds)")
	recognizer := sherpa.NewOnlineRecognizer(&config)
	log.Println("Recognizer created!")
	defer sherpa.DeleteOnlineRecognizer(recognizer)

	log.Println("Start decoding!")
	stream := sherpa.NewOnlineStream(recognizer)
	defer sherpa.DeleteOnlineStream(stream)

	stream.AcceptWaveform(wav.SampleRate, samplesf32)

	tailPadding := make([]float32, int(float32(wav.SampleRate)*0.3))
	stream.AcceptWaveform(wav.SampleRate, tailPadding)

	for recognizer.IsReady(stream) {
		recognizer.Decode(stream)
	}
	log.Println("Decoding done!")
	result := recognizer.GetResult(stream)
	log.Println(result.Text)
}

func samplesInt16ToFloat(inSamples []byte) []float32 {
	numSamples := len(inSamples) / 2
	outSamples := make([]float32, numSamples)

	for i := 0; i != numSamples; i++ {
		s := inSamples[i*2 : (i+1)*2]

		var s16 int16
		buf := bytes.NewReader(s)
		err := binary.Read(buf, binary.LittleEndian, &s16)
		if err != nil {
			log.Fatal("Failed to parse 16-bit sample")
		}
		outSamples[i] = float32(s16) / 32768
	}

	return outSamples
}
