package main

import (
	"log"

	sherpa "github.com/k2-fsa/sherpa-onnx-go/sherpa_onnx"
)

func main() {
	log.SetFlags(log.LstdFlags | log.Lmicroseconds)

	config := sherpa.OnlinePunctuationConfig{}
	config.Model.CnnBilstm = "./sherpa-onnx-online-punct-en-2024-08-06/model.onnx"
	config.Model.BpeVocab = "./sherpa-onnx-online-punct-en-2024-08-06/bpe.vocab"
	config.Model.NumThreads = 1
	config.Model.Provider = "cpu"

	punct := sherpa.NewOnlinePunctuation(&config)
	if punct == nil {
		log.Fatal("Failed to create OnlinePunctuation")
	}
	defer sherpa.DeleteOnlinePunctuation(punct)

	textArray := []string{
		"how are you i am fine thank you",
		"The African blogosphere is rapidly expanding bringing more voices online in the form of commentaries opinions analyses rants and poetry",
	}

	log.Println("----------")
	for _, text := range textArray {
		newText := punct.AddPunct(text)
		log.Printf("Input text: %v", text)
		log.Printf("Output text: %v", newText)
		log.Println("----------")
	}
}
