package main

import (
	"fmt"
	portaudio "github.com/csukuangfj/portaudio-go"
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
	config.NumThreads = 2
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

func createVad() *sherpa.VoiceActivityDetector {
	config := sherpa.VadModelConfig{}

	// Please download silero_vad.onnx from
	// https://github.com/snakers4/silero-vad/blob/master/files/silero_vad.onnx

	config.SileroVad.Model = "./silero_vad.onnx"
	config.SileroVad.Threshold = 0.5
	config.SileroVad.MinSilenceDuration = 0.5
	config.SileroVad.MinSpeechDuration = 0.5
	config.SileroVad.WindowSize = 512
	config.SampleRate = 16000
	config.NumThreads = 1
	config.Provider = "cpu"
	config.Debug = 1

	var bufferSizeInSeconds float32 = 20

	vad := sherpa.NewVoiceActivityDetector(&config, bufferSizeInSeconds)
	return vad
}

func main() {
	log.SetFlags(log.LstdFlags | log.Lmicroseconds)

	vad := createVad()
	defer sherpa.DeleteVoiceActivityDetector(vad)

	ex := createSpeakerEmbeddingExtractor()
	defer sherpa.DeleteSpeakerEmbeddingExtractor(ex)

	manager := sherpa.NewSpeakerEmbeddingManager(ex.Dim())
	defer sherpa.DeleteSpeakerEmbeddingManager(manager)
	registerSpeakers(ex, manager)

	err := portaudio.Initialize()
	if err != nil {
		log.Fatalf("Unable to initialize portaudio: %v\n", err)
	}
	defer portaudio.Terminate()

	default_device, err := portaudio.DefaultInputDevice()
	if err != nil {
		log.Fatal("Failed to get default input device: %v\n", err)
	}
	log.Printf("Selected default input device: %s\n", default_device.Name)
	param := portaudio.StreamParameters{}
	param.Input.Device = default_device
	param.Input.Channels = 1
	param.Input.Latency = default_device.DefaultHighInputLatency

	param.SampleRate = 16000
	param.FramesPerBuffer = 0
	param.Flags = portaudio.ClipOff

	// you can choose another value for 0.1 if you want
	samplesPerCall := int32(param.SampleRate * 0.1) // 0.1 second
	samples := make([]float32, samplesPerCall)

	s, err := portaudio.OpenStream(param, samples)
	if err != nil {
		log.Fatalf("Failed to open the stream")
	}

	defer s.Close()
	chk(s.Start())

	log.Print("Started! Please speak")
	printed := false

	k := 0
	for {
		chk(s.Read())
		vad.AcceptWaveform(samples)

		if vad.IsSpeech() && !printed {
			printed = true
			log.Print("Detected speech\n")
		}

		if !vad.IsSpeech() {
			printed = false
		}

		for !vad.IsEmpty() {
			speechSegment := vad.Front()
			vad.Pop()

			audio := &sherpa.Wave{}
			audio.Samples = speechSegment.Samples
			audio.SampleRate = 16000

			// Now decode it
			go decode(ex, manager, audio, k)

			k += 1
		}
	}

	chk(s.Stop())

}

func chk(err error) {
	if err != nil {
		panic(err)
	}
}

func decode(ex *sherpa.SpeakerEmbeddingExtractor, manager *sherpa.SpeakerEmbeddingManager, audio *sherpa.GeneratedAudio, id int) {
	stream := ex.CreateStream()
	defer sherpa.DeleteOnlineStream(stream)

	stream.AcceptWaveform(audio.SampleRate, audio.Samples)
	stream.InputFinished()
	embeddings := ex.Compute(stream)
	threshold := float32(0.5)
	name := manager.Search(embeddings, threshold)
	if len(name) > 0 {
		log.Printf("Found speaker: %v\n", name)
	} else {
		log.Print("Unknown speaker\n")
		name = "Unknown"
	}

	duration := float32(len(audio.Samples)) / float32(audio.SampleRate)

	filename := fmt.Sprintf("seg-%d-%.2f-seconds-%s.wav", id, duration, name)
	ok := audio.Save(filename)
	if ok {
		log.Printf("Saved to %s", filename)
	}
	log.Print("----------\n")
}
