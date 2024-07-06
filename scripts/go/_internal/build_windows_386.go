//go:build windows && 386

package sherpa_onnx

// #cgo LDFLAGS: -L ${SRCDIR}/lib/i686-pc-windows-gnu -lsherpa-onnx-c-api -lonnxruntime
import "C"
