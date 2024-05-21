//go:build darwin && arm64 && !ios

package sherpa_onnx

// #cgo LDFLAGS: -L ${SRCDIR}/lib/aarch64-apple-darwin -lsherpa-onnx-c-api -lsherpa-onnx-core -lkaldi-native-fbank-core  -lkaldi-decoder-core -lsherpa-onnx-kaldifst-core -lsherpa-onnx-fstfar -lsherpa-onnx-fst -lpiper_phonemize -lespeak-ng -lucd -lonnxruntime -Wl,-rpath,${SRCDIR}/lib/aarch64-apple-darwin
import "C"
