//go:build windows && amd64

package sherpa_onnx

// #cgo LDFLAGS: -L ${SRCDIR}/lib/x86_64-pc-windows-gnu -lsherpa-onnx-c-api -lonnxruntime
import "C"
