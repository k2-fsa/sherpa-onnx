package main

import (
	"github.com/gordonklaus/portaudio"
	sherpa "github.com/k2-fsa/sherpa-onnx-go/sherpa_onnx"
	"log"
)

func main() {
	err := portaudio.Initialize()
	if err != nil {
		log.Fatalf("Unable to initialize portaudio: %v\n", err)
	}
	defer portaudio.Terminate()
	log.Println(portaudio.VersionText())

	default_device, err := portaudio.DefaultInputDevice()
	if err != nil {
		log.Fatal("Failed to get default input device: %v\n", err)
	}
	log.Printf("Select default input device: %s\n", default_device.Name)
	param := portaudio.StreamParameters{}
	param.Input.Device = default_device
	param.Input.Channels = 1
	param.Input.Latency = default_device.DefaultLowInputLatency

	param.SampleRate = 16000
	param.FramesPerBuffer = 0
	param.Flags = portaudio.ClipOff

	config := sherpa.OnlineRecognizerConfig{}
	config.FeatConfig = sherpa.FeatureConfig{SampleRate: 16000, FeatureDim: 80}
	config.ModelConfig = sherpa.OnlineTransducerModelConfig{
		// Encoder:    "./sherpa-onnx-streaming-zipformer-en-2023-06-26/encoder-epoch-99-avg-1-chunk-16-left-128.onnx",
		// Decoder:    "./sherpa-onnx-streaming-zipformer-en-2023-06-26/decoder-epoch-99-avg-1-chunk-16-left-128.onnx",
		// Joiner:     "./sherpa-onnx-streaming-zipformer-en-2023-06-26/joiner-epoch-99-avg-1-chunk-16-left-128.onnx",
		// Tokens:     "./sherpa-onnx-streaming-zipformer-en-2023-06-26/tokens.txt",
		Encoder:    "./icefall-asr-zipformer-streaming-wenetspeech-20230615/exp/encoder-epoch-12-avg-4-chunk-16-left-128.onnx",
		Decoder:    "./icefall-asr-zipformer-streaming-wenetspeech-20230615/exp/decoder-epoch-12-avg-4-chunk-16-left-128.onnx",
		Joiner:     "./icefall-asr-zipformer-streaming-wenetspeech-20230615/exp/joiner-epoch-12-avg-4-chunk-16-left-128.onnx",
		Tokens:     "./icefall-asr-zipformer-streaming-wenetspeech-20230615/data/lang_char/tokens.txt",
		NumThreads: 1,
		Provider:   "cpu",
		Debug:      1,
		ModelType:  "zipformer2",
	}
	config.DecodingMethod = "greedy_search"
	config.EnableEndpoint = 0

	recognizer := sherpa.NewOnlineRecognizer(&config)
	defer sherpa.DeleteOnlineRecognizer(recognizer)

	stream := sherpa.NewOnlineStream(recognizer)

	// you can choose another value for 0.1 if you want
	samplesPerCall := int32(param.SampleRate * 0.1) // 0.1 second

	samples := make([]float32, samplesPerCall)
	s, err := portaudio.OpenStream(param, samples)
	if err != nil {
		log.Fatalf("Failed to open the stream")
	}
	defer s.Close()
	chk(s.Start())

	for {
		chk(s.Read())
		stream.AcceptWaveform(int(param.SampleRate), samples)

		for recognizer.IsReady(stream) {
			recognizer.Decode(stream)
		}
		log.Println(recognizer.GetResult(stream).Text)
	}

	chk(s.Stop())
	return

}

func chk(err error) {
	if err != nil {
		panic(err)
	}
}
