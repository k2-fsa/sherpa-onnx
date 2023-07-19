package sherpa_onnx

// #include <stdlib.h>
// #include "c-api.h"
import "C"
import "unsafe"

type OnlineTransducerModelConfig struct {
	Encoder    string
	Decoder    string
	Joiner     string
	Tokens     string
	NumThreads int
	Provider   string
	Debug      int
	ModelType  string
}

type FeatureConfig struct {
	SampleRate int
	FeatureDim int
}

type OnlineRecognizerConfig struct {
	FeatConfig              FeatureConfig
	ModelConfig             OnlineTransducerModelConfig
	DecodingMethod          string
	MaxActivePaths          int
	EnableEndpoint          int
	Rule1MinTrailingSilence float32
	Rule2MinTrailingSilence float32
	Rule3MinUtteranceLength float32
}

type OnlineRecognizerResult struct {
	Text string
}

type OnlineRecognizer struct {
	impl *C.struct_SherpaOnnxOnlineRecognizer
}

type OnlineStream struct {
	impl *C.struct_SherpaOnnxOnlineStream
}

func DeleteOnlineRecognizer(recognizer *OnlineRecognizer) {
	C.DestroyOnlineRecognizer(recognizer.impl)
	recognizer.impl = nil
}

// The user is responsible to invoke DeleteOnlineRecognizer() to free
// the returned recognizer to avoid memory leak
func NewOnlineRecognizer(config *OnlineRecognizerConfig) *OnlineRecognizer {
	c := C.struct_SherpaOnnxOnlineRecognizerConfig{}
	c.feat_config.sample_rate = C.int(config.FeatConfig.SampleRate)
	c.feat_config.feature_dim = C.int(config.FeatConfig.FeatureDim)

	c.model_config.encoder = C.CString(config.ModelConfig.Encoder)
	defer C.free(unsafe.Pointer(c.model_config.encoder))

	c.model_config.decoder = C.CString(config.ModelConfig.Decoder)
	defer C.free(unsafe.Pointer(c.model_config.decoder))

	c.model_config.joiner = C.CString(config.ModelConfig.Joiner)
	defer C.free(unsafe.Pointer(c.model_config.joiner))

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

func DeleteOnlineStream(stream *OnlineStream) {
	C.DestroyOnlineStream(stream.impl)
	stream.impl = nil
}

// The user is responsible to invoke DeleteOnlineStream() to free
// the returned stream to avoid memory leak
func NewOnlineStream(recognizer *OnlineRecognizer) *OnlineStream {
	stream := &OnlineStream{}
	stream.impl = C.CreateOnlineStream(recognizer.impl)
	return stream
}

func (s *OnlineStream) AcceptWaveform(sampleRate int, samples []float32) {
	C.AcceptWaveform(s.impl, C.int(sampleRate), (*C.float)(&samples[0]), C.int(len(samples)))
}

func (s *OnlineStream) InputFinished() {
	C.InputFinished(s.impl)
}

func (recognizer *OnlineRecognizer) IsReady(s *OnlineStream) bool {
	return C.IsOnlineStreamReady(recognizer.impl, s.impl) == 1
}

func (recognizer *OnlineRecognizer) IsEndpoint(s *OnlineStream) bool {
	return C.IsEndpoint(recognizer.impl, s.impl) == 1
}

func (recognizer *OnlineRecognizer) Reset(s *OnlineStream) {
	C.Reset(recognizer.impl, s.impl)
}

func (recognizer *OnlineRecognizer) Decode(s *OnlineStream) {
	C.DecodeOnlineStream(recognizer.impl, s.impl)
}

func (recognizer *OnlineRecognizer) DecodeStreams(s []*OnlineStream) {
	ss := make([]*C.struct_SherpaOnnxOnlineStream, len(s))
	for i, v := range s {
		ss[i] = v.impl
	}

	C.DecodeMultipleOnlineStreams(recognizer.impl, &ss[0], C.int(len(s)))
}

func (recognizer *OnlineRecognizer) GetResult(s *OnlineStream) *OnlineRecognizerResult {
	p := C.GetOnlineStreamResult(recognizer.impl, s.impl)
	defer C.DestroyOnlineRecognizerResult(p)
	result := &OnlineRecognizerResult{}
	result.Text = C.GoString(p.text)

	return result
}
