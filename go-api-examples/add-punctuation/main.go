package main

import (
	sherpa "github.com/k2-fsa/sherpa-onnx-go/sherpa_onnx"
	"log"
)

func main() {
	log.SetFlags(log.LstdFlags | log.Lmicroseconds)

	config := sherpa.OfflinePunctuationConfig{}
	config.Model.CtTransformer = "./sherpa-onnx-punct-ct-transformer-zh-en-vocab272727-2024-04-12/model.onnx"
	config.Model.NumThreads = 1
	config.Model.Provider = "cpu"

	punct := sherpa.NewOfflinePunctuation(&config)
	defer sherpa.DeleteOfflinePunc(punct)

	textArray := []string{
		"这是一个测试你好吗How are you我很好thank you are you ok谢谢你",
		"我们都是木头人不会说话不会动",
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
