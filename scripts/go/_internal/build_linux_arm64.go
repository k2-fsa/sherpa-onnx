//go:build linux && arm64

package sherpa_onnx

// #cgo LDFLAGS: -L ${SRCDIR}/lib/aarch64-unknown-linux-gnu -lsherpa-onnx-c-api -lsherpa-onnx-core -lkaldi-native-fbank-core  -lkaldi-decoder-core -lsherpa-onnx-kaldifst-core -lsherpa-onnx-fstfar -lsherpa-onnx-fst -lpiper_phonemize -lespeak-ng -lucd -lonnxruntime -lssentencepiece_core -Wl,-rpath,${SRCDIR}/lib/aarch64-unknown-linux-gnu
import "C"
