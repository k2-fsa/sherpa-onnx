//go:build !android && linux && amd64 && !musl

package sherpa_onnx

// #cgo LDFLAGS: -L ${SRCDIR}/lib/x86_64-unknown-linux-gnu -lsherpa-onnx-c-api -lonnxruntime -Wl,-rpath,${SRCDIR}/lib/x86_64-unknown-linux-gnu
import "C"
