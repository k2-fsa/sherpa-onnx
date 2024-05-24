//go:build darwin && amd64 && !ios

package sherpa_onnx

// #cgo LDFLAGS: -L ${SRCDIR}/lib/x86_64-apple-darwin -lsherpa-onnx-c-api -lsherpa-onnx-core -lkaldi-native-fbank-core  -lkaldi-decoder-core -lsherpa-onnx-kaldifst-core -lsherpa-onnx-fstfar -lsherpa-onnx-fst -lpiper_phonemize -lespeak-ng -lucd -lonnxruntime -lssentencepiece_core -Wl,-rpath,${SRCDIR}/lib/x86_64-apple-darwin
import "C"
