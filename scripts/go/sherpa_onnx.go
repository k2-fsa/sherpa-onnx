/*
Speech recognition with [Next-gen Kaldi].

[sherpa-onnx] is an open-source speech recognition framework for [Next-gen Kaldi].
It depends only on [onnxruntime], supporting both streaming and non-streaming
speech recognition.

It does not need to access the network during recognition and everything
runs locally.

It supports a variety of platforms, such as Linux (x86_64, aarch64, arm),
Windows (x86_64, x86), macOS (x86_64, arm64), etc.

Usage examples:

 1. Real-time speech recognition from a microphone

    Please see
    https://github.com/k2-fsa/sherpa-onnx/tree/master/go-api-examples/real-time-speech-recognition-from-microphone

 2. Decode files using a non-streaming model

    Please see
    https://github.com/k2-fsa/sherpa-onnx/tree/master/go-api-examples/non-streaming-decode-files

 3. Decode files using a streaming model

    Please see
    https://github.com/k2-fsa/sherpa-onnx/tree/master/go-api-examples/streaming-decode-files

[sherpa-onnx]: https://github.com/k2-fsa/sherpa-onnx
[onnxruntime]: https://github.com/microsoft/onnxruntime
[Next-gen Kaldi]: https://github.com/k2-fsa/
*/
package sherpa_onnx

// #include <stdlib.h>
// #include "c-api.h"
import "C"
import "unsafe"

// Configuration for online/streaming transducer models
//
// Please refer to
// https://k2-fsa.github.io/sherpa/onnx/pretrained_models/online-transducer/index.html
// to download pre-trained models
type OnlineTransducerModelConfig struct {
	Encoder string // Path to the encoder model, e.g., encoder.onnx or encoder.int8.onnx
	Decoder string // Path to the decoder model.
	Joiner  string // Path to the joiner model.
}

// Configuration for online/streaming paraformer models
//
// Please refer to
// https://k2-fsa.github.io/sherpa/onnx/pretrained_models/online-paraformer/index.html
// to download pre-trained models
type OnlineParaformerModelConfig struct {
	Encoder string // Path to the encoder model, e.g., encoder.onnx or encoder.int8.onnx
	Decoder string // Path to the decoder model.
}

// Configuration for online/streaming models
//
// Please refer to
// https://k2-fsa.github.io/sherpa/onnx/pretrained_models/online-transducer/index.html
// https://k2-fsa.github.io/sherpa/onnx/pretrained_models/online-paraformer/index.html
// to download pre-trained models
type OnlineModelConfig struct {
	Transducer OnlineTransducerModelConfig
	Paraformer OnlineParaformerModelConfig
	Tokens     string // Path to tokens.txt
	NumThreads int    // Number of threads to use for neural network computation
	Provider   string // Optional. Valid values are: cpu, cuda, coreml
	Debug      int    // 1 to show model meta information while loading it.
	ModelType  string // Optional. You can specify it for faster model initialization
}

// Configuration for the feature extractor
type FeatureConfig struct {
	// Sample rate expected by the model. It is 16000 for all
	// pre-trained models provided by us
	SampleRate int
	// Feature dimension expected by the model. It is 80 for all
	// pre-trained models provided by us
	FeatureDim int
}

// Configuration for the online/streaming recognizer.
type OnlineRecognizerConfig struct {
	FeatConfig  FeatureConfig
	ModelConfig OnlineModelConfig

	// Valid decoding methods: greedy_search, modified_beam_search
	DecodingMethod string

	// Used only when DecodingMethod is modified_beam_search. It specifies
	// the maximum number of paths to keep during the search
	MaxActivePaths int

	EnableEndpoint int // 1 to enable endpoint detection.

	// Please see
	// https://k2-fsa.github.io/sherpa/ncnn/endpoint.html
	// for the meaning of Rule1MinTrailingSilence, Rule2MinTrailingSilence
	// and Rule3MinUtteranceLength.
	Rule1MinTrailingSilence float32
	Rule2MinTrailingSilence float32
	Rule3MinUtteranceLength float32
}

// It contains the recognition result for a online stream.
type OnlineRecognizerResult struct {
	Text string
}

// The online recognizer class. It wraps a pointer from C.
type OnlineRecognizer struct {
	impl *C.struct_SherpaOnnxOnlineRecognizer
}

// The online stream class. It wraps a pointer from C.
type OnlineStream struct {
	impl *C.struct_SherpaOnnxOnlineStream
}

// Free the internal pointer inside the recognizer to avoid memory leak.
func DeleteOnlineRecognizer(recognizer *OnlineRecognizer) {
	C.DestroyOnlineRecognizer(recognizer.impl)
	recognizer.impl = nil
}

// The user is responsible to invoke [DeleteOnlineRecognizer]() to free
// the returned recognizer to avoid memory leak
func NewOnlineRecognizer(config *OnlineRecognizerConfig) *OnlineRecognizer {
	c := C.struct_SherpaOnnxOnlineRecognizerConfig{}
	c.feat_config.sample_rate = C.int(config.FeatConfig.SampleRate)
	c.feat_config.feature_dim = C.int(config.FeatConfig.FeatureDim)

	c.model_config.transducer.encoder = C.CString(config.ModelConfig.Transducer.Encoder)
	defer C.free(unsafe.Pointer(c.model_config.transducer.encoder))

	c.model_config.transducer.decoder = C.CString(config.ModelConfig.Transducer.Decoder)
	defer C.free(unsafe.Pointer(c.model_config.transducer.decoder))

	c.model_config.transducer.joiner = C.CString(config.ModelConfig.Transducer.Joiner)
	defer C.free(unsafe.Pointer(c.model_config.transducer.joiner))

	c.model_config.paraformer.encoder = C.CString(config.ModelConfig.Paraformer.Encoder)
	defer C.free(unsafe.Pointer(c.model_config.paraformer.encoder))

	c.model_config.paraformer.decoder = C.CString(config.ModelConfig.Paraformer.Decoder)
	defer C.free(unsafe.Pointer(c.model_config.paraformer.decoder))

	c.model_config.tokens = C.CString(config.ModelConfig.Tokens)
	defer C.free(unsafe.Pointer(c.model_config.tokens))

	c.model_config.num_threads = C.int(config.ModelConfig.NumThreads)

	c.model_config.provider = C.CString(config.ModelConfig.Provider)
	defer C.free(unsafe.Pointer(c.model_config.provider))

	c.model_config.debug = C.int(config.ModelConfig.Debug)

	c.model_config.model_type = C.CString(config.ModelConfig.ModelType)
	defer C.free(unsafe.Pointer(c.model_config.model_type))

	c.decoding_method = C.CString(config.DecodingMethod)
	defer C.free(unsafe.Pointer(c.decoding_method))

	c.max_active_paths = C.int(config.MaxActivePaths)
	c.enable_endpoint = C.int(config.EnableEndpoint)
	c.rule1_min_trailing_silence = C.float(config.Rule1MinTrailingSilence)
	c.rule2_min_trailing_silence = C.float(config.Rule2MinTrailingSilence)
	c.rule3_min_utterance_length = C.float(config.Rule3MinUtteranceLength)

	recognizer := &OnlineRecognizer{}
	recognizer.impl = C.CreateOnlineRecognizer(&c)

	return recognizer
}

// Delete the internal pointer inside the stream to avoid memory leak.
func DeleteOnlineStream(stream *OnlineStream) {
	C.DestroyOnlineStream(stream.impl)
	stream.impl = nil
}

// The user is responsible to invoke [DeleteOnlineStream]() to free
// the returned stream to avoid memory leak
func NewOnlineStream(recognizer *OnlineRecognizer) *OnlineStream {
	stream := &OnlineStream{}
	stream.impl = C.CreateOnlineStream(recognizer.impl)
	return stream
}

// Input audio samples for the stream.
//
// sampleRate is the actual sample rate of the input audio samples. If it
// is different from the sample rate expected by the feature extractor, we will
// do resampling inside.
//
// samples contains audio samples. Each sample is in the range [-1, 1]
func (s *OnlineStream) AcceptWaveform(sampleRate int, samples []float32) {
	C.AcceptWaveform(s.impl, C.int(sampleRate), (*C.float)(&samples[0]), C.int(len(samples)))
}

// Signal that there will be no incoming audio samples.
// After calling this function, you cannot call [OnlineStream.AcceptWaveform] any longer.
//
// The main purpose of this function is to flush the remaining audio samples
// buffered inside for feature extraction.
func (s *OnlineStream) InputFinished() {
	C.InputFinished(s.impl)
}

// Check whether the stream has enough feature frames for decoding.
// Return true if this stream is ready for decoding. Return false otherwise.
//
// You will usually use it like below:
//
//	for recognizer.IsReady(s) {
//	   recognizer.Decode(s)
//	}
func (recognizer *OnlineRecognizer) IsReady(s *OnlineStream) bool {
	return C.IsOnlineStreamReady(recognizer.impl, s.impl) == 1
}

// Return true if an endpoint is detected.
//
// You usually use it like below:
//
//	if recognizer.IsEndpoint(s) {
//	   // do your own stuff after detecting an endpoint
//
//	   recognizer.Reset(s)
//	}
func (recognizer *OnlineRecognizer) IsEndpoint(s *OnlineStream) bool {
	return C.IsEndpoint(recognizer.impl, s.impl) == 1
}

// After calling this function, the internal neural network model states
// are reset and IsEndpoint(s) would return false. GetResult(s) would also
// return an empty string.
func (recognizer *OnlineRecognizer) Reset(s *OnlineStream) {
	C.Reset(recognizer.impl, s.impl)
}

// Decode the stream. Before calling this function, you have to ensure
// that recognizer.IsReady(s) returns true. Otherwise, you will be SAD.
//
// You usually use it like below:
//
//	for recognizer.IsReady(s) {
//	  recognizer.Decode(s)
//	}
func (recognizer *OnlineRecognizer) Decode(s *OnlineStream) {
	C.DecodeOnlineStream(recognizer.impl, s.impl)
}

// Decode multiple streams in parallel, i.e., in batch.
// You have to ensure that each stream is ready for decoding. Otherwise,
// you will be SAD.
func (recognizer *OnlineRecognizer) DecodeStreams(s []*OnlineStream) {
	ss := make([]*C.struct_SherpaOnnxOnlineStream, len(s))
	for i, v := range s {
		ss[i] = v.impl
	}

	C.DecodeMultipleOnlineStreams(recognizer.impl, &ss[0], C.int(len(s)))
}

// Get the current result of stream since the last invoke of Reset()
func (recognizer *OnlineRecognizer) GetResult(s *OnlineStream) *OnlineRecognizerResult {
	p := C.GetOnlineStreamResult(recognizer.impl, s.impl)
	defer C.DestroyOnlineRecognizerResult(p)
	result := &OnlineRecognizerResult{}
	result.Text = C.GoString(p.text)

	return result
}

// Configuration for offline/non-streaming transducer.
//
// Please refer to
// https://k2-fsa.github.io/sherpa/onnx/pretrained_models/offline-transducer/index.html
// to download pre-trained models
type OfflineTransducerModelConfig struct {
	Encoder string // Path to the encoder model, i.e., encoder.onnx or encoder.int8.onnx
	Decoder string // Path to the decoder model
	Joiner  string // Path to the joiner model
}

// Configuration for offline/non-streaming paraformer.
//
// please refer to
// https://k2-fsa.github.io/sherpa/onnx/pretrained_models/offline-paraformer/index.html
// to download pre-trained models
type OfflineParaformerModelConfig struct {
	Model string // Path to the model, e.g., model.onnx or model.int8.onnx
}

// Configuration for offline/non-streaming NeMo CTC models.
//
// Please refer to
// https://k2-fsa.github.io/sherpa/onnx/pretrained_models/offline-ctc/index.html
// to download pre-trained models
type OfflineNemoEncDecCtcModelConfig struct {
	Model string // Path to the model, e.g., model.onnx or model.int8.onnx
}

type OfflineWhisperModelConfig struct {
	Encoder string
	Decoder string
}

type OfflineTdnnModelConfig struct {
	Model string
}

// Configuration for offline LM.
type OfflineLMConfig struct {
	Model string  // Path to the model
	Scale float32 // scale for LM score
}

type OfflineModelConfig struct {
	Transducer OfflineTransducerModelConfig
	Paraformer OfflineParaformerModelConfig
	NemoCTC    OfflineNemoEncDecCtcModelConfig
	Whisper    OfflineWhisperModelConfig
	Tdnn       OfflineTdnnModelConfig
	Tokens     string // Path to tokens.txt

	// Number of threads to use for neural network computation
	NumThreads int

	// 1 to print model meta information while loading
	Debug int

	// Optional. Valid values: cpu, cuda, coreml
	Provider string

	// Optional. Specify it for faster model initialization.
	ModelType string
}

// Configuration for the offline/non-streaming recognizer.
type OfflineRecognizerConfig struct {
	FeatConfig  FeatureConfig
	ModelConfig OfflineModelConfig
	LmConfig    OfflineLMConfig

	// Valid decoding method: greedy_search, modified_beam_search
	DecodingMethod string

	// Used only when DecodingMethod is modified_beam_search.
	MaxActivePaths int
}

// It wraps a pointer from C
type OfflineRecognizer struct {
	impl *C.struct_SherpaOnnxOfflineRecognizer
}

// It wraps a pointer from C
type OfflineStream struct {
	impl *C.struct_SherpaOnnxOfflineStream
}

// It contains recognition result of an offline stream.
type OfflineRecognizerResult struct {
	Text string
}

// Frees the internal pointer of the recognition to avoid memory leak.
func DeleteOfflineRecognizer(recognizer *OfflineRecognizer) {
	C.DestroyOfflineRecognizer(recognizer.impl)
	recognizer.impl = nil
}

// The user is responsible to invoke [DeleteOfflineRecognizer]() to free
// the returned recognizer to avoid memory leak
func NewOfflineRecognizer(config *OfflineRecognizerConfig) *OfflineRecognizer {
	c := C.struct_SherpaOnnxOfflineRecognizerConfig{}
	c.feat_config.sample_rate = C.int(config.FeatConfig.SampleRate)
	c.feat_config.feature_dim = C.int(config.FeatConfig.FeatureDim)

	c.model_config.transducer.encoder = C.CString(config.ModelConfig.Transducer.Encoder)
	defer C.free(unsafe.Pointer(c.model_config.transducer.encoder))

	c.model_config.transducer.decoder = C.CString(config.ModelConfig.Transducer.Decoder)
	defer C.free(unsafe.Pointer(c.model_config.transducer.decoder))

	c.model_config.transducer.joiner = C.CString(config.ModelConfig.Transducer.Joiner)
	defer C.free(unsafe.Pointer(c.model_config.transducer.joiner))

	c.model_config.paraformer.model = C.CString(config.ModelConfig.Paraformer.Model)
	defer C.free(unsafe.Pointer(c.model_config.paraformer.model))

	c.model_config.nemo_ctc.model = C.CString(config.ModelConfig.NemoCTC.Model)
	defer C.free(unsafe.Pointer(c.model_config.nemo_ctc.model))

	c.model_config.whisper.encoder = C.CString(config.ModelConfig.Whisper.Encoder)
	defer C.free(unsafe.Pointer(c.model_config.whisper.encoder))

	c.model_config.whisper.decoder = C.CString(config.ModelConfig.Whisper.Decoder)
	defer C.free(unsafe.Pointer(c.model_config.whisper.decoder))

	c.model_config.tdnn.decoder = C.CString(config.ModelConfig.Tdnn.Model)
	defer C.free(unsafe.Pointer(c.model_config.tdnn.model))

	c.model_config.tokens = C.CString(config.ModelConfig.Tokens)
	defer C.free(unsafe.Pointer(c.model_config.tokens))

	c.model_config.num_threads = C.int(config.ModelConfig.NumThreads)

	c.model_config.debug = C.int(config.ModelConfig.Debug)

	c.model_config.provider = C.CString(config.ModelConfig.Provider)
	defer C.free(unsafe.Pointer(c.model_config.provider))

	c.model_config.model_type = C.CString(config.ModelConfig.ModelType)
	defer C.free(unsafe.Pointer(c.model_config.model_type))

	c.lm_config.model = C.CString(config.LmConfig.Model)
	defer C.free(unsafe.Pointer(c.lm_config.model))

	c.lm_config.scale = C.float(config.LmConfig.Scale)

	c.decoding_method = C.CString(config.DecodingMethod)
	defer C.free(unsafe.Pointer(c.decoding_method))

	c.max_active_paths = C.int(config.MaxActivePaths)

	recognizer := &OfflineRecognizer{}
	recognizer.impl = C.CreateOfflineRecognizer(&c)

	return recognizer
}

// Frees the internal pointer of the stream to avoid memory leak.
func DeleteOfflineStream(stream *OfflineStream) {
	C.DestroyOfflineStream(stream.impl)
	stream.impl = nil
}

// The user is responsible to invoke [DeleteOfflineStream]() to free
// the returned stream to avoid memory leak
func NewOfflineStream(recognizer *OfflineRecognizer) *OfflineStream {
	stream := &OfflineStream{}
	stream.impl = C.CreateOfflineStream(recognizer.impl)
	return stream
}

// Input audio samples for the offline stream.
// Please only call it once. That is, input all samples at once.
//
// sampleRate is the sample rate of the input audio samples. If it is different
// from the value expected by the feature extractor, we will do resampling inside.
//
// samples contains the actual audio samples. Each sample is in the range [-1, 1].
func (s *OfflineStream) AcceptWaveform(sampleRate int, samples []float32) {
	C.AcceptWaveformOffline(s.impl, C.int(sampleRate), (*C.float)(&samples[0]), C.int(len(samples)))
}

// Decode the offline stream.
func (recognizer *OfflineRecognizer) Decode(s *OfflineStream) {
	C.DecodeOfflineStream(recognizer.impl, s.impl)
}

// Decode multiple streams in parallel, i.e., in batch.
func (recognizer *OfflineRecognizer) DecodeStreams(s []*OfflineStream) {
	ss := make([]*C.struct_SherpaOnnxOfflineStream, len(s))
	for i, v := range s {
		ss[i] = v.impl
	}

	C.DecodeMultipleOfflineStreams(recognizer.impl, &ss[0], C.int(len(s)))
}

// Get the recognition result of the offline stream.
func (s *OfflineStream) GetResult() *OfflineRecognizerResult {
	p := C.GetOfflineStreamResult(s.impl)
	defer C.DestroyOfflineRecognizerResult(p)
	result := &OfflineRecognizerResult{}
	result.Text = C.GoString(p.text)

	return result
}
