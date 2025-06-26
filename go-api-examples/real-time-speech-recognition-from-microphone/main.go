package main

import (
	"bytes"
	"encoding/binary"
	"fmt"
	"github.com/gen2brain/malgo"
	sherpa "github.com/k2-fsa/sherpa-onnx-go/sherpa_onnx"
	flag "github.com/spf13/pflag"
	"log"
	"strings"
)

func initRecognizer() *sherpa.OnlineRecognizer {
	config := sherpa.OnlineRecognizerConfig{}
	config.FeatConfig = sherpa.FeatureConfig{SampleRate: 16000, FeatureDim: 80}

	flag.StringVar(&config.ModelConfig.Transducer.Encoder, "encoder", "", "Path to the transducer encoder model")
	flag.StringVar(&config.ModelConfig.Transducer.Decoder, "decoder", "", "Path to the transducer decoder model")
	flag.StringVar(&config.ModelConfig.Transducer.Joiner, "joiner", "", "Path to the transducer joiner model")
	flag.StringVar(&config.ModelConfig.Paraformer.Encoder, "paraformer-encoder", "", "Path to the paraformer encoder model")
	flag.StringVar(&config.ModelConfig.Paraformer.Decoder, "paraformer-decoder", "", "Path to the paraformer decoder model")
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
	return recognizer
}

func main() {
	ctx, err := malgo.InitContext(nil, malgo.ContextConfig{}, func(message string) {
		fmt.Printf("LOG <%v>", message)
	})
	chk(err)

	defer func() {
		_ = ctx.Uninit()
		ctx.Free()
	}()

	deviceConfig := malgo.DefaultDeviceConfig(malgo.Duplex)
	deviceConfig.Capture.Format = malgo.FormatS16
	deviceConfig.Capture.Channels = 1
	deviceConfig.Playback.Format = malgo.FormatS16
	deviceConfig.Playback.Channels = 1
	deviceConfig.SampleRate = 16000
	deviceConfig.Alsa.NoMMap = 1

	recognizer := initRecognizer()
	defer sherpa.DeleteOnlineRecognizer(recognizer)

	stream := sherpa.NewOnlineStream(recognizer)
	defer sherpa.DeleteOnlineStream(stream)

	var last_text string

	segment_idx := 0

	onRecvFrames := func(_, pSample []byte, framecount uint32) {
		samples := samplesInt16ToFloat(pSample)
		stream.AcceptWaveform(16000, samples)

		// Please use a separate goroutine for decoding in your app
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

	captureCallbacks := malgo.DeviceCallbacks{
		Data: onRecvFrames,
	}

	device, err := malgo.InitDevice(ctx.Context, deviceConfig, captureCallbacks)
	chk(err)

	err = device.Start()
	chk(err)
	fmt.Println("Started. Please speak. Press ctrl + C  to exit")
	fmt.Scanln()
	device.Uninit()
}

func chk(err error) {
	if err != nil {
		panic(err)
	}
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
