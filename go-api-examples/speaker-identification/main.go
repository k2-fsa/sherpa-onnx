package main

import (
	sherpa "github.com/k2-fsa/sherpa-onnx-go/sherpa_onnx"
	"log"
)

func createSpeakerEmbeddingExtractor() *sherpa.SpeakerEmbeddingExtractor {
	config := sherpa.SpeakerEmbeddingExtractorConfig{}

	// Please download the model from
	// https://github.com/k2-fsa/sherpa-onnx/releases/download/speaker-recongition-models/3dspeaker_speech_campplus_sv_zh-cn_16k-common.onnx
	//
	// You can find more models at
	// https://github.com/k2-fsa/sherpa-onnx/releases/tag/speaker-recongition-models

	config.Model = "./3dspeaker_speech_campplus_sv_zh-cn_16k-common.onnx"
	config.NumThreads = 1
	config.Debug = 1
	config.Provider = "cpu"

	ex := sherpa.NewSpeakerEmbeddingExtractor(&config)
	return ex
}

func computeEmbeddings(ex *sherpa.SpeakerEmbeddingExtractor, files []string) [][]float32 {
	embeddings := make([][]float32, len(files))

	for i, f := range files {
		wave := sherpa.ReadWave(f)

		stream := ex.CreateStream()
		defer sherpa.DeleteOnlineStream(stream)
		stream.AcceptWaveform(wave.SampleRate, wave.Samples)
		stream.InputFinished()
		embeddings[i] = ex.Compute(stream)
	}

	return embeddings

}

func registerSpeakers(ex *sherpa.SpeakerEmbeddingExtractor, manager *sherpa.SpeakerEmbeddingManager) {
	// Please download the test data from
	// https://github.com/csukuangfj/sr-data
	spk1_files := []string{
		"./sr-data/enroll/fangjun-sr-1.wav",
		"./sr-data/enroll/fangjun-sr-2.wav",
		"./sr-data/enroll/fangjun-sr-3.wav",
	}

	spk2_files := []string{
		"./sr-data/enroll/leijun-sr-1.wav",
		"./sr-data/enroll/leijun-sr-2.wav",
	}

	spk1_embeddings := computeEmbeddings(ex, spk1_files)
	spk2_embeddings := computeEmbeddings(ex, spk2_files)

	ok := manager.RegisterV("fangjun", spk1_embeddings)
	if !ok {
		panic("Failed to register fangjun")
	}

	ok = manager.RegisterV("leijun", spk2_embeddings)
	if !ok {
		panic("Failed to register leijun")
	}

	if !manager.Contains("fangjun") {
		panic("Failed to find fangjun")
	}

	if !manager.Contains("leijun") {
		panic("Failed to find leijun")
	}

	if manager.NumSpeakers() != 2 {
		panic("There should be only 2 speakers")
	}

	all_speakers := manager.AllSpeakers()
	log.Printf("All speakers: %v\n", all_speakers)
}

func main() {
	log.SetFlags(log.LstdFlags | log.Lmicroseconds)

	ex := createSpeakerEmbeddingExtractor()
	defer sherpa.DeleteSpeakerEmbeddingExtractor(ex)

	manager := sherpa.NewSpeakerEmbeddingManager(ex.Dim())
	defer sherpa.DeleteSpeakerEmbeddingManager(manager)
	registerSpeakers(ex, manager)

	// Please download the test data from
	// https://github.com/csukuangfj/sr-data
	test1 := "./sr-data/test/fangjun-test-sr-1.wav"
	embeddings := computeEmbeddings(ex, []string{test1})[0]
	threshold := float32(0.6)
	name := manager.Search(embeddings, threshold)
	if len(name) > 0 {
		log.Printf("%v matches %v", test1, name)
	} else {
		log.Printf("No matches found for %v", test1)
	}

	test2 := "./sr-data/test/leijun-test-sr-1.wav"
	embeddings = computeEmbeddings(ex, []string{test2})[0]
	name = manager.Search(embeddings, threshold)
	if len(name) > 0 {
		log.Printf("%v matches %v", test2, name)
	} else {
		log.Printf("No matches found for %v", test2)
	}

	test3 := "./sr-data/test/liudehua-test-sr-1.wav"
	embeddings = computeEmbeddings(ex, []string{test3})[0]
	name = manager.Search(embeddings, threshold)
	if len(name) > 0 {
		log.Printf("%v matches %v", test3, name)
	} else {
		log.Printf("No matches found for %v", test3)
	}

	if !manager.Remove("fangjun") {
		panic("Failed to deregister fangjun")
	} else {
		log.Print("fangjun deregistered\n")
	}

	test1 = "./sr-data/test/fangjun-test-sr-1.wav"
	embeddings = computeEmbeddings(ex, []string{test1})[0]
	name = manager.Search(embeddings, threshold)
	if len(name) > 0 {
		log.Printf("%v matches %v", test1, name)
	} else {
		log.Printf("No matches found for %v", test1)
	}
}

func chk(err error) {
	if err != nil {
		panic(err)
	}
}
