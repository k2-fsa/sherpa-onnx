//go:build darwin && amd64 && !ios

package main

// #cgo LDFLAGS: -L ${SRCDIR}/lib/x86_64-apple-darwin -lsherpa-onnx-c-api -lsherpa-onnx-core -lkaldi-native-fbank-core -lonnxruntime -Wl,-rpath,${SRCDIR}/lib/x86_64-apple-darwin
import "C"
