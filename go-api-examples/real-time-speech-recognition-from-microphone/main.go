package main

import (
	"fmt"
	"github.com/gordonklaus/portaudio"
	sherpa "github.com/k2-fsa/sherpa-onnx-go/sherpa_onnx"
	flag "github.com/spf13/pflag"
	"log"
	"strings"
)

func main() {
	err := portaudio.Initialize()
	if err != nil {
		log.Fatalf("Unable to initialize portaudio: %v\n", err)
	}
	defer portaudio.Terminate()

	default_device, err := portaudio.DefaultInputDevice()
	if err != nil {
		log.Fatal("Failed to get default input device: %v\n", err)
	}
	fmt.Printf("Select default input device: %s\n", default_device.Name)
	param := portaudio.StreamParameters{}
	param.Input.Device = default_device
	param.Input.Channels = 1
	param.Input.Latency = default_device.DefaultLowInputLatency

	param.SampleRate = 16000
	param.FramesPerBuffer = 0
	param.Flags = portaudio.ClipOff

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
	flag.IntVar(&config.EnableEndpoint, "enable-endpoint", 1, "Whether to enable endpoint")
	flag.Float32Var(&config.Rule1MinTrailingSilence, "rule1-min-trailing-silence", 2.4, "Threshold for rule1")
	flag.Float32Var(&config.Rule2MinTrailingSilence, "rule2-min-trailing-silence", 1.2, "Threshold for rule2")
	flag.Float32Var(&config.Rule3MinUtteranceLength, "rule3-min-utterance-length", 20, "Threshold for rule3")

	flag.Parse()

	log.Println("Initializing recognizer (may take several seconds)")
	recognizer := sherpa.NewOnlineRecognizer(&config)
	log.Println("Recognizer created!")
	defer sherpa.DeleteOnlineRecognizer(recognizer)

	stream := sherpa.NewOnlineStream(recognizer)
	defer sherpa.NewOfflineStream(stream)

	// you can choose another value for 0.1 if you want
	samplesPerCall := int32(param.SampleRate * 0.1) // 0.1 second

	samples := make([]float32, samplesPerCall)
	s, err := portaudio.OpenStream(param, samples)
	if err != nil {
		log.Fatalf("Failed to open the stream")
	}
	defer s.Close()
	chk(s.Start())

	var last_text string

	segment_idx := 0

	fmt.Println("Started! Please speak")

	for {
		chk(s.Read())
		stream.AcceptWaveform(int(param.SampleRate), samples)

		for recognizer.IsReady(stream) {
			recognizer.Decode(stream)
		}

		text := recognizer.GetResult(stream).Text
		if len(text) != 0 && last_text != text {
			last_text = strings.ToLower(text)
			fmt.Printf("\r%d: %s", segment_idx, last_text)
		}

		if recognizer.IsEndpoint(stream) {
			if len(text) != 0 {
				segment_idx++
				fmt.Println()
			}
			recognizer.Reset(stream)
		}
	}

	chk(s.Stop())
}

func chk(err error) {
	if err != nil {
		panic(err)
	}
}
