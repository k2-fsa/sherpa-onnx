package main

import (
	"fmt"
	"log"
	"os"
	"time"

	sherpa "github.com/k2-fsa/sherpa-onnx-go/sherpa_onnx"
)

func main() {
	log.SetFlags(log.LstdFlags | log.Lmicroseconds)

	modelType := "spleeter"
	if len(os.Args) > 1 {
		modelType = os.Args[1]
	}

	config := sherpa.OfflineSourceSeparationConfig{}
	config.Model.NumThreads = 1
	config.Model.Debug = true

	var stemNames []string

	switch modelType {
	case "spleeter":
		modelDir := "./sherpa-onnx-spleeter-2stems-fp16"
		config.Model.Spleeter = sherpa.OfflineSourceSeparationSpleeterModelConfig{
			Vocals:        modelDir + "/vocals.fp16.onnx",
			Accompaniment: modelDir + "/accompaniment.fp16.onnx",
		}
		stemNames = []string{"vocals", "accompaniment"}
	case "uvr":
		config.Model.Uvr = sherpa.OfflineSourceSeparationUvrModelConfig{
			Model: "./UVR-MDX-NET-Voc_FT.onnx",
		}
		stemNames = []string{"uvr-vocals", "uvr-non-vocals"}
	default:
		log.Fatalf("Unknown model type: %s. Use 'spleeter' or 'uvr'.\n", modelType)
	}

	separator := sherpa.NewSourceSeparator(config)
	if separator == nil {
		log.Fatal("Failed to create SourceSeparator")
	}
	defer separator.Delete()

	inputFile := "./qi-feng-le-zh.wav"
	input := sherpa.ReadWaveMultiChannel(inputFile)
	if input == nil {
		log.Fatalf("Failed to read %s\n", inputFile)
	}
	defer input.Release()

	fmt.Printf("Input: channels=%d, samples=%d, sampleRate=%d\n",
		input.ChannelCount, input.SamplesPerChannel, input.SampleRate)

	start := time.Now()
	stems := separator.Process(input)
	elapsed := time.Since(start).Seconds()
	audioDuration := float64(input.SamplesPerChannel) / float64(input.SampleRate)
	rtf := elapsed / audioDuration

	fmt.Printf("Output: %d stems, sampleRate=%d\n", len(stems), input.SampleRate)
	fmt.Printf("Elapsed: %.2fs, Audio: %.2fs, RTF = %.3f\n", elapsed, audioDuration, rtf)

	for i := 0; i < len(stems) && i < len(stemNames); i++ {
		filename := stemNames[i] + ".wav"
		ok := stems[i].Save(filename)
		if ok {
			fmt.Printf("Saved %s\n", filename)
		} else {
			fmt.Printf("Failed to save %s\n", filename)
		}
	}
}
