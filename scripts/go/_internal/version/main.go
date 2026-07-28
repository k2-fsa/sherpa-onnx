package main

import (
	"log"

	sherpa "github.com/k2-fsa/sherpa-onnx-go/sherpa_onnx"
)

func main() {
	log.Printf("sherpa-onnx version: %v\n", sherpa.GetVersion())
	log.Printf("sherpa-onnx gitSha1: %v\n", sherpa.GetGitSha1())
	log.Printf("sherpa-onnx gitDate: %v\n", sherpa.GetGitDate())
	log.Printf("onnxruntime version: %v\n", sherpa.GetOnnxruntimeVersion())
}
