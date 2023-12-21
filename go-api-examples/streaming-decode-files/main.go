package main

import (
	"bytes"
	"encoding/binary"
	sherpa "github.com/k2-fsa/sherpa-onnx-go/sherpa_onnx"
	flag "github.com/spf13/pflag"
	"github.com/youpy/go-wav"
	"log"
	"os"
	"strings"
)

func main() {
	log.SetFlags(log.LstdFlags | log.Lmicroseconds)

	config := sherpa.OnlineRecognizerConfig{}
	config.FeatConfig = sherpa.FeatureConfig{SampleRate: 16000, FeatureDim: 80}

	flag.StringVar(&config.ModelConfig.Transducer.Encoder, "encoder", "", "Path to the transducer encoder model")
	flag.StringVar(&config.ModelConfig.Transducer.Decoder, "decoder", "", "Path to the transducer decoder model")
	flag.StringVar(&config.ModelConfig.Transducer.Joiner, "joiner", "", "Path to the transducer joiner model")
	flag.StringVar(&config.ModelConfig.Paraformer.Encoder, "paraformer-encoder", "", "Path to the paraformer encoder model")
	flag.StringVar(&config.ModelConfig.Paraformer.Decoder, "paraformer-decoder", "", "Path to the paraformer decoder model")
	flag.StringVar(&config.ModelConfig.Zipformer2Ctc.Model, "zipformer2-ctc", "", "Path to the zipformer2 CTC model")
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

	samples, sampleRate := readWave(flag.Arg(0))

	log.Println("Initializing recognizer (may take several seconds)")
	recognizer := sherpa.NewOnlineRecognizer(&config)
	log.Println("Recognizer created!")
	defer sherpa.DeleteOnlineRecognizer(recognizer)

	log.Println("Start decoding!")
	stream := sherpa.NewOnlineStream(recognizer)
	defer sherpa.DeleteOnlineStream(stream)

	stream.AcceptWaveform(sampleRate, samples)

	tailPadding := make([]float32, int(float32(sampleRate)*0.3))
	stream.AcceptWaveform(sampleRate, tailPadding)

	for recognizer.IsReady(stream) {
		recognizer.Decode(stream)
	}
	log.Println("Decoding done!")
	result := recognizer.GetResult(stream)
	log.Println(strings.ToLower(result.Text))
	log.Printf("Wave duration: %v seconds", float32(len(samples))/float32(sampleRate))
}

func readWave(filename string) (samples []float32, sampleRate int) {
	file, _ := os.Open(filename)
	defer file.Close()

	reader := wav.NewReader(file)
	format, err := reader.Format()
	if err != nil {
		log.Fatalf("Failed to read wave format")
	}

	if format.AudioFormat != 1 {
		log.Fatalf("Support only PCM format. Given: %v\n", format.AudioFormat)
	}

	if format.NumChannels != 1 {
		log.Fatalf("Support only 1 channel wave file. Given: %v\n", format.NumChannels)
	}

	if format.BitsPerSample != 16 {
		log.Fatalf("Support only 16-bit per sample. Given: %v\n", format.BitsPerSample)
	}

	reader.Duration() // so that it initializes reader.Size

	buf := make([]byte, reader.Size)
	n, err := reader.Read(buf)
	if n != int(reader.Size) {
		log.Fatalf("Failed to read %v bytes. Returned %v bytes\n", reader.Size, n)
	}

	samples = samplesInt16ToFloat(buf)
	sampleRate = int(format.SampleRate)

	return
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
