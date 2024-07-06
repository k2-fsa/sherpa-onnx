//go:build linux && arm && !arm7

package sherpa_onnx

// #cgo LDFLAGS: -L ${SRCDIR}/lib/arm-unknown-linux-gnueabihf -lsherpa-onnx-c-api -lonnxruntime -Wl,-rpath,${SRCDIR}/lib/arm-unknown-linux-gnueabihf
import "C"
