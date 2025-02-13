package main

import (
	"fmt"
	sherpa "github.com/k2-fsa/sherpa-onnx-go/sherpa_onnx"
	"log"
)

func main() {
	config := sherpa.AudioTaggingConfig{}
	config.Model.Zipformer.Model = "./sherpa-onnx-zipformer-small-audio-tagging-2024-04-15/model.int8.onnx"
	config.Model.NumThreads = 1
	config.Model.Debug = 1
	config.Model.Provider = "cpu"
	config.Labels = "./sherpa-onnx-zipformer-small-audio-tagging-2024-04-15/class_labels_indices.csv"
	config.TopK = 5

	tagging := sherpa.NewAudioTagging(&config)
	defer sherpa.DeleteAudioTagging(tagging)

	wave_filename := "./sherpa-onnx-zipformer-small-audio-tagging-2024-04-15/test_wavs/3.wav"

	wave := sherpa.ReadWave(wave_filename)
	if wave == nil {
		log.Printf("Failed to read %v\n", wave_filename)
		return
	}

	stream := sherpa.NewAudioTaggingStream(tagging)
	defer sherpa.DeleteOfflineStream(stream)

	stream.AcceptWaveform(wave.SampleRate, wave.Samples)

	result := tagging.Compute(stream, 10)
	fmt.Printf("the tagging result: %v\n", result)
}
