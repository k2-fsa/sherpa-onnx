//go:build darwin && arm64 && !ios

package sherpa_onnx

// #cgo LDFLAGS: -L ${SRCDIR}/lib/aarch64-apple-darwin -lsherpa-onnx-c-api -lonnxruntime -lssentencepiece_core -Wl,-rpath,${SRCDIR}/lib/aarch64-apple-darwin
import "C"
