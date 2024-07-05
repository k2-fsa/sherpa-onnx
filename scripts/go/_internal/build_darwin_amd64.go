//go:build darwin && amd64 && !ios

package sherpa_onnx

// #cgo LDFLAGS: -L ${SRCDIR}/lib/x86_64-apple-darwin -lsherpa-onnx-c-api -lonnxruntime -Wl,-rpath,${SRCDIR}/lib/x86_64-apple-darwin
import "C"
