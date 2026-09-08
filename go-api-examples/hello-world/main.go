package main

import (
	"fmt"

	sherpa "github.com/k2-fsa/sherpa-onnx-go/sherpa_onnx"
)

func main() {
	fmt.Printf("sherpa-onnx version: %s\n", sherpa.GetVersion())
	fmt.Printf("sherpa-onnx gitSha1: %s\n", sherpa.GetGitSha1())
	fmt.Printf("sherpa-onnx gitDate: %s\n", sherpa.GetGitDate())
	fmt.Printf("onnxruntime version: %s\n", sherpa.GetOnnxruntimeVersion())
}
