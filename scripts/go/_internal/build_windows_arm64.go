//go:build windows && arm64

package sherpa_onnx

// #cgo LDFLAGS: -L ${SRCDIR}/lib/aarch64-pc-windows-gnu -lsherpa-onnx-c-api -lonnxruntime
import "C"
