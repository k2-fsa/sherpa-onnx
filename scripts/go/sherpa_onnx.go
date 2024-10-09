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

 4. Convert text to speech using a non-streaming model

    Please see
    https://github.com/k2-fsa/sherpa-onnx/tree/master/go-api-examples/non-streaming-tts

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

// Please refer to
// https://k2-fsa.github.io/sherpa/onnx/pretrained_models/online-ctc/index.html
// to download pre-trained models
type OnlineZipformer2CtcModelConfig struct {
	Model string // Path to the onnx model
}

// Configuration for online/streaming models
//
// Please refer to
// https://k2-fsa.github.io/sherpa/onnx/pretrained_models/online-transducer/index.html
// https://k2-fsa.github.io/sherpa/onnx/pretrained_models/online-paraformer/index.html
// to download pre-trained models
type OnlineModelConfig struct {
	Transducer    OnlineTransducerModelConfig
	Paraformer    OnlineParaformerModelConfig
	Zipformer2Ctc OnlineZipformer2CtcModelConfig
	Tokens        string // Path to tokens.txt
	NumThreads    int    // Number of threads to use for neural network computation
	Provider      string // Optional. Valid values are: cpu, cuda, coreml
	Debug         int    // 1 to show model meta information while loading it.
	ModelType     string // Optional. You can specify it for faster model initialization
	ModelingUnit  string // Optional. cjkchar, bpe, cjkchar+bpe
	BpeVocab      string // Optional.
	TokensBuf     string // Optional.
	TokensBufSize int    // Optional.
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

type OnlineCtcFstDecoderConfig struct {
	Graph     string
	MaxActive int
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
	HotwordsFile            string
	HotwordsScore           float32
	BlankPenalty            float32
	CtcFstDecoderConfig     OnlineCtcFstDecoderConfig
	RuleFsts                string
	RuleFars                string
	HotwordsBuf             string
	HotwordsBufSize         int
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
	C.SherpaOnnxDestroyOnlineRecognizer(recognizer.impl)
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

	c.model_config.zipformer2_ctc.model = C.CString(config.ModelConfig.Zipformer2Ctc.Model)
	defer C.free(unsafe.Pointer(c.model_config.zipformer2_ctc.model))

	c.model_config.tokens = C.CString(config.ModelConfig.Tokens)
	defer C.free(unsafe.Pointer(c.model_config.tokens))

	c.model_config.tokens_buf = C.CString(config.ModelConfig.TokensBuf)
	defer C.free(unsafe.Pointer(c.model_config.tokens_buf))

	c.model_config.tokens_buf_size = C.int(config.ModelConfig.TokensBufSize)

	c.model_config.num_threads = C.int(config.ModelConfig.NumThreads)

	c.model_config.provider = C.CString(config.ModelConfig.Provider)
	defer C.free(unsafe.Pointer(c.model_config.provider))

	c.model_config.debug = C.int(config.ModelConfig.Debug)

	c.model_config.model_type = C.CString(config.ModelConfig.ModelType)
	defer C.free(unsafe.Pointer(c.model_config.model_type))

	c.model_config.modeling_unit = C.CString(config.ModelConfig.ModelingUnit)
	defer C.free(unsafe.Pointer(c.model_config.modeling_unit))

	c.model_config.bpe_vocab = C.CString(config.ModelConfig.BpeVocab)
	defer C.free(unsafe.Pointer(c.model_config.bpe_vocab))

	c.decoding_method = C.CString(config.DecodingMethod)
	defer C.free(unsafe.Pointer(c.decoding_method))

	c.max_active_paths = C.int(config.MaxActivePaths)
	c.enable_endpoint = C.int(config.EnableEndpoint)
	c.rule1_min_trailing_silence = C.float(config.Rule1MinTrailingSilence)
	c.rule2_min_trailing_silence = C.float(config.Rule2MinTrailingSilence)
	c.rule3_min_utterance_length = C.float(config.Rule3MinUtteranceLength)

	c.hotwords_file = C.CString(config.HotwordsFile)
	defer C.free(unsafe.Pointer(c.hotwords_file))

	c.hotwords_buf = C.CString(config.HotwordsBuf)
	defer C.free(unsafe.Pointer(c.hotwords_buf))

	c.hotwords_buf_size = C.int(config.HotwordsBufSize)

	c.hotwords_score = C.float(config.HotwordsScore)
	c.blank_penalty = C.float(config.BlankPenalty)

	c.rule_fsts = C.CString(config.RuleFsts)
	defer C.free(unsafe.Pointer(c.rule_fsts))

	c.rule_fars = C.CString(config.RuleFars)
	defer C.free(unsafe.Pointer(c.rule_fars))

	c.ctc_fst_decoder_config.graph = C.CString(config.CtcFstDecoderConfig.Graph)
	defer C.free(unsafe.Pointer(c.ctc_fst_decoder_config.graph))
	c.ctc_fst_decoder_config.max_active = C.int(config.CtcFstDecoderConfig.MaxActive)

	recognizer := &OnlineRecognizer{}
	recognizer.impl = C.SherpaOnnxCreateOnlineRecognizer(&c)

	return recognizer
}

// Delete the internal pointer inside the stream to avoid memory leak.
func DeleteOnlineStream(stream *OnlineStream) {
	C.SherpaOnnxDestroyOnlineStream(stream.impl)
	stream.impl = nil
}

// The user is responsible to invoke [DeleteOnlineStream]() to free
// the returned stream to avoid memory leak
func NewOnlineStream(recognizer *OnlineRecognizer) *OnlineStream {
	stream := &OnlineStream{}
	stream.impl = C.SherpaOnnxCreateOnlineStream(recognizer.impl)
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
	C.SherpaOnnxOnlineStreamAcceptWaveform(s.impl, C.int(sampleRate), (*C.float)(&samples[0]), C.int(len(samples)))
}

// Signal that there will be no incoming audio samples.
// After calling this function, you cannot call [OnlineStream.AcceptWaveform] any longer.
//
// The main purpose of this function is to flush the remaining audio samples
// buffered inside for feature extraction.
func (s *OnlineStream) InputFinished() {
	C.SherpaOnnxOnlineStreamInputFinished(s.impl)
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
	return C.SherpaOnnxIsOnlineStreamReady(recognizer.impl, s.impl) == 1
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
	return C.SherpaOnnxOnlineStreamIsEndpoint(recognizer.impl, s.impl) == 1
}

// After calling this function, the internal neural network model states
// are reset and IsEndpoint(s) would return false. GetResult(s) would also
// return an empty string.
func (recognizer *OnlineRecognizer) Reset(s *OnlineStream) {
	C.SherpaOnnxOnlineStreamReset(recognizer.impl, s.impl)
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
	C.SherpaOnnxDecodeOnlineStream(recognizer.impl, s.impl)
}

// Decode multiple streams in parallel, i.e., in batch.
// You have to ensure that each stream is ready for decoding. Otherwise,
// you will be SAD.
func (recognizer *OnlineRecognizer) DecodeStreams(s []*OnlineStream) {
	ss := make([]*C.struct_SherpaOnnxOnlineStream, len(s))
	for i, v := range s {
		ss[i] = v.impl
	}

	C.SherpaOnnxDecodeMultipleOnlineStreams(recognizer.impl, &ss[0], C.int(len(s)))
}

// Get the current result of stream since the last invoke of Reset()
func (recognizer *OnlineRecognizer) GetResult(s *OnlineStream) *OnlineRecognizerResult {
	p := C.SherpaOnnxGetOnlineStreamResult(recognizer.impl, s.impl)
	defer C.SherpaOnnxDestroyOnlineRecognizerResult(p)
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
	Encoder      string
	Decoder      string
	Language     string
	Task         string
	TailPaddings int
}

type OfflineTdnnModelConfig struct {
	Model string
}

type OfflineSenseVoiceModelConfig struct {
	Model                       string
	Language                    string
	UseInverseTextNormalization int
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
	SenseVoice OfflineSenseVoiceModelConfig
	Tokens     string // Path to tokens.txt

	// Number of threads to use for neural network computation
	NumThreads int

	// 1 to print model meta information while loading
	Debug int

	// Optional. Valid values: cpu, cuda, coreml
	Provider string

	// Optional. Specify it for faster model initialization.
	ModelType string

	ModelingUnit  string // Optional. cjkchar, bpe, cjkchar+bpe
	BpeVocab      string // Optional.
	TeleSpeechCtc string // Optional.
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
	HotwordsFile   string
	HotwordsScore  float32
	BlankPenalty   float32
	RuleFsts       string
	RuleFars       string
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
	Text       string
	Tokens     []string
	Timestamps []float32
	Lang       string
	Emotion    string
	Event      string
}

// Frees the internal pointer of the recognition to avoid memory leak.
func DeleteOfflineRecognizer(recognizer *OfflineRecognizer) {
	C.SherpaOnnxDestroyOfflineRecognizer(recognizer.impl)
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

	c.model_config.whisper.language = C.CString(config.ModelConfig.Whisper.Language)
	defer C.free(unsafe.Pointer(c.model_config.whisper.language))

	c.model_config.whisper.task = C.CString(config.ModelConfig.Whisper.Task)
	defer C.free(unsafe.Pointer(c.model_config.whisper.task))

	c.model_config.whisper.tail_paddings = C.int(config.ModelConfig.Whisper.TailPaddings)

	c.model_config.tdnn.model = C.CString(config.ModelConfig.Tdnn.Model)
	defer C.free(unsafe.Pointer(c.model_config.tdnn.model))

	c.model_config.sense_voice.model = C.CString(config.ModelConfig.SenseVoice.Model)
	defer C.free(unsafe.Pointer(c.model_config.sense_voice.model))

	c.model_config.sense_voice.language = C.CString(config.ModelConfig.SenseVoice.Language)
	defer C.free(unsafe.Pointer(c.model_config.sense_voice.language))

	c.model_config.sense_voice.use_itn = C.int(config.ModelConfig.SenseVoice.UseInverseTextNormalization)

	c.model_config.tokens = C.CString(config.ModelConfig.Tokens)
	defer C.free(unsafe.Pointer(c.model_config.tokens))

	c.model_config.num_threads = C.int(config.ModelConfig.NumThreads)

	c.model_config.debug = C.int(config.ModelConfig.Debug)

	c.model_config.provider = C.CString(config.ModelConfig.Provider)
	defer C.free(unsafe.Pointer(c.model_config.provider))

	c.model_config.model_type = C.CString(config.ModelConfig.ModelType)
	defer C.free(unsafe.Pointer(c.model_config.model_type))

	c.model_config.modeling_unit = C.CString(config.ModelConfig.ModelingUnit)
	defer C.free(unsafe.Pointer(c.model_config.modeling_unit))

	c.model_config.bpe_vocab = C.CString(config.ModelConfig.BpeVocab)
	defer C.free(unsafe.Pointer(c.model_config.bpe_vocab))

	c.model_config.telespeech_ctc = C.CString(config.ModelConfig.TeleSpeechCtc)
	defer C.free(unsafe.Pointer(c.model_config.telespeech_ctc))

	c.lm_config.model = C.CString(config.LmConfig.Model)
	defer C.free(unsafe.Pointer(c.lm_config.model))

	c.lm_config.scale = C.float(config.LmConfig.Scale)

	c.decoding_method = C.CString(config.DecodingMethod)
	defer C.free(unsafe.Pointer(c.decoding_method))

	c.max_active_paths = C.int(config.MaxActivePaths)

	c.hotwords_file = C.CString(config.HotwordsFile)
	defer C.free(unsafe.Pointer(c.hotwords_file))

	c.hotwords_score = C.float(config.HotwordsScore)

	c.blank_penalty = C.float(config.BlankPenalty)

	c.rule_fsts = C.CString(config.RuleFsts)
	defer C.free(unsafe.Pointer(c.rule_fsts))

	c.rule_fars = C.CString(config.RuleFars)
	defer C.free(unsafe.Pointer(c.rule_fars))

	recognizer := &OfflineRecognizer{}
	recognizer.impl = C.SherpaOnnxCreateOfflineRecognizer(&c)

	return recognizer
}

// Frees the internal pointer of the stream to avoid memory leak.
func DeleteOfflineStream(stream *OfflineStream) {
	C.SherpaOnnxDestroyOfflineStream(stream.impl)
	stream.impl = nil
}

// The user is responsible to invoke [DeleteOfflineStream]() to free
// the returned stream to avoid memory leak
func NewOfflineStream(recognizer *OfflineRecognizer) *OfflineStream {
	stream := &OfflineStream{}
	stream.impl = C.SherpaOnnxCreateOfflineStream(recognizer.impl)
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
	C.SherpaOnnxAcceptWaveformOffline(s.impl, C.int(sampleRate), (*C.float)(&samples[0]), C.int(len(samples)))
}

// Decode the offline stream.
func (recognizer *OfflineRecognizer) Decode(s *OfflineStream) {
	C.SherpaOnnxDecodeOfflineStream(recognizer.impl, s.impl)
}

// Decode multiple streams in parallel, i.e., in batch.
func (recognizer *OfflineRecognizer) DecodeStreams(s []*OfflineStream) {
	ss := make([]*C.struct_SherpaOnnxOfflineStream, len(s))
	for i, v := range s {
		ss[i] = v.impl
	}

	C.SherpaOnnxDecodeMultipleOfflineStreams(recognizer.impl, &ss[0], C.int(len(s)))
}

// Get the recognition result of the offline stream.
func (s *OfflineStream) GetResult() *OfflineRecognizerResult {
	p := C.SherpaOnnxGetOfflineStreamResult(s.impl)
	defer C.SherpaOnnxDestroyOfflineRecognizerResult(p)
	n := int(p.count)
	if n == 0 {
		return nil
	}
	result := &OfflineRecognizerResult{}
	result.Text = C.GoString(p.text)
	result.Lang = C.GoString(p.lang)
	result.Emotion = C.GoString(p.emotion)
	result.Event = C.GoString(p.event)
	result.Tokens = make([]string, n)
	tokens := (*[1 << 28]*C.char)(unsafe.Pointer(p.tokens_arr))[:n:n]
	for i := 0; i < n; i++ {
		result.Tokens[i] = C.GoString(tokens[i])
	}
	if p.timestamps == nil {
		return result
	}
	result.Timestamps = make([]float32, n)
	timestamps := (*[1 << 28]C.float)(unsafe.Pointer(p.timestamps))[:n:n]
	for i := 0; i < n; i++ {
		result.Timestamps[i] = float32(timestamps[i])
	}
	return result
}

// Configuration for offline/non-streaming text-to-speech (TTS).
//
// Please refer to
// https://k2-fsa.github.io/sherpa/onnx/tts/pretrained_models/index.html
// to download pre-trained models
type OfflineTtsVitsModelConfig struct {
	Model       string  // Path to the VITS onnx model
	Lexicon     string  // Path to lexicon.txt
	Tokens      string  // Path to tokens.txt
	DataDir     string  // Path to espeak-ng-data directory
	NoiseScale  float32 // noise scale for vits models. Please use 0.667 in general
	NoiseScaleW float32 // noise scale for vits models. Please use 0.8 in general
	LengthScale float32 // Please use 1.0 in general. Smaller -> Faster speech speed. Larger -> Slower speech speed
	DictDir     string  // Path to dict directory for jieba (used only in Chinese tts)
}

type OfflineTtsModelConfig struct {
	Vits OfflineTtsVitsModelConfig

	// Number of threads to use for neural network computation
	NumThreads int

	// 1 to print model meta information while loading
	Debug int

	// Optional. Valid values: cpu, cuda, coreml
	Provider string
}

type OfflineTtsConfig struct {
	Model           OfflineTtsModelConfig
	RuleFsts        string
	RuleFars        string
	MaxNumSentences int
}

type GeneratedAudio struct {
	// Normalized samples in the range [-1, 1]
	Samples []float32

	SampleRate int
}

// The offline tts class. It wraps a pointer from C.
type OfflineTts struct {
	impl *C.struct_SherpaOnnxOfflineTts
}

// Free the internal pointer inside the tts to avoid memory leak.
func DeleteOfflineTts(tts *OfflineTts) {
	C.SherpaOnnxDestroyOfflineTts(tts.impl)
	tts.impl = nil
}

// The user is responsible to invoke [DeleteOfflineTts]() to free
// the returned tts to avoid memory leak
func NewOfflineTts(config *OfflineTtsConfig) *OfflineTts {
	c := C.struct_SherpaOnnxOfflineTtsConfig{}

	c.rule_fsts = C.CString(config.RuleFsts)
	defer C.free(unsafe.Pointer(c.rule_fsts))

	c.rule_fars = C.CString(config.RuleFars)
	defer C.free(unsafe.Pointer(c.rule_fars))

	c.max_num_sentences = C.int(config.MaxNumSentences)

	c.model.vits.model = C.CString(config.Model.Vits.Model)
	defer C.free(unsafe.Pointer(c.model.vits.model))

	c.model.vits.lexicon = C.CString(config.Model.Vits.Lexicon)
	defer C.free(unsafe.Pointer(c.model.vits.lexicon))

	c.model.vits.tokens = C.CString(config.Model.Vits.Tokens)
	defer C.free(unsafe.Pointer(c.model.vits.tokens))

	c.model.vits.data_dir = C.CString(config.Model.Vits.DataDir)
	defer C.free(unsafe.Pointer(c.model.vits.data_dir))

	c.model.vits.noise_scale = C.float(config.Model.Vits.NoiseScale)
	c.model.vits.noise_scale_w = C.float(config.Model.Vits.NoiseScaleW)
	c.model.vits.length_scale = C.float(config.Model.Vits.LengthScale)

	c.model.vits.dict_dir = C.CString(config.Model.Vits.DictDir)
	defer C.free(unsafe.Pointer(c.model.vits.dict_dir))

	c.model.num_threads = C.int(config.Model.NumThreads)
	c.model.debug = C.int(config.Model.Debug)

	c.model.provider = C.CString(config.Model.Provider)
	defer C.free(unsafe.Pointer(c.model.provider))

	tts := &OfflineTts{}
	tts.impl = C.SherpaOnnxCreateOfflineTts(&c)

	return tts
}

func (tts *OfflineTts) Generate(text string, sid int, speed float32) *GeneratedAudio {
	s := C.CString(text)
	defer C.free(unsafe.Pointer(s))

	audio := C.SherpaOnnxOfflineTtsGenerate(tts.impl, s, C.int(sid), C.float(speed))
	defer C.SherpaOnnxDestroyOfflineTtsGeneratedAudio(audio)

	ans := &GeneratedAudio{}
	ans.SampleRate = int(audio.sample_rate)
	n := int(audio.n)
	ans.Samples = make([]float32, n)

	// see https://stackoverflow.com/questions/48756732/what-does-1-30c-yourtype-do-exactly-in-cgo
	// :n:n means 0:n:n, means low:high:capacity
	samples := (*[1 << 28]C.float)(unsafe.Pointer(audio.samples))[:n:n]
	// copy(ans.Samples, samples)
	for i := 0; i < n; i++ {
		ans.Samples[i] = float32(samples[i])
	}

	return ans
}

func (audio *GeneratedAudio) Save(filename string) bool {
	s := C.CString(filename)
	defer C.free(unsafe.Pointer(s))

	ok := int(C.SherpaOnnxWriteWave((*C.float)(&audio.Samples[0]), C.int(len(audio.Samples)), C.int(audio.SampleRate), s))

	return ok == 1
}

// ============================================================
// For VAD
// ============================================================
type SileroVadModelConfig struct {
	Model              string
	Threshold          float32
	MinSilenceDuration float32
	MinSpeechDuration  float32
	WindowSize         int
	MaxSpeechDuration  float32
}

type VadModelConfig struct {
	SileroVad  SileroVadModelConfig
	SampleRate int
	NumThreads int
	Provider   string
	Debug      int
}

type CircularBuffer struct {
	impl *C.struct_SherpaOnnxCircularBuffer
}

func DeleteCircularBuffer(buffer *CircularBuffer) {
	C.SherpaOnnxDestroyCircularBuffer(buffer.impl)
	buffer.impl = nil
}

func NewCircularBuffer(capacity int) *CircularBuffer {
	circularBuffer := &CircularBuffer{}
	circularBuffer.impl = C.SherpaOnnxCreateCircularBuffer(C.int(capacity))
	return circularBuffer
}

func (buffer *CircularBuffer) Push(samples []float32) {
	C.SherpaOnnxCircularBufferPush(buffer.impl, (*C.float)(&samples[0]), C.int(len(samples)))
}

func (buffer *CircularBuffer) Get(start int, n int) []float32 {
	samples := C.SherpaOnnxCircularBufferGet(buffer.impl, C.int(start), C.int(n))
	defer C.SherpaOnnxCircularBufferFree(samples)

	result := make([]float32, n)

	p := (*[1 << 28]C.float)(unsafe.Pointer(samples))[:n:n]
	for i := 0; i < n; i++ {
		result[i] = float32(p[i])
	}

	return result
}

func (buffer *CircularBuffer) Pop(n int) {
	C.SherpaOnnxCircularBufferPop(buffer.impl, C.int(n))
}

func (buffer *CircularBuffer) Size() int {
	return int(C.SherpaOnnxCircularBufferSize(buffer.impl))
}

func (buffer *CircularBuffer) Head() int {
	return int(C.SherpaOnnxCircularBufferHead(buffer.impl))
}

func (buffer *CircularBuffer) Reset() {
	C.SherpaOnnxCircularBufferReset(buffer.impl)
}

type SpeechSegment struct {
	Start   int
	Samples []float32
}

type VoiceActivityDetector struct {
	impl *C.struct_SherpaOnnxVoiceActivityDetector
}

func NewVoiceActivityDetector(config *VadModelConfig, bufferSizeInSeconds float32) *VoiceActivityDetector {
	c := C.struct_SherpaOnnxVadModelConfig{}

	c.silero_vad.model = C.CString(config.SileroVad.Model)
	defer C.free(unsafe.Pointer(c.silero_vad.model))

	c.silero_vad.threshold = C.float(config.SileroVad.Threshold)
	c.silero_vad.min_silence_duration = C.float(config.SileroVad.MinSilenceDuration)
	c.silero_vad.min_speech_duration = C.float(config.SileroVad.MinSpeechDuration)
	c.silero_vad.window_size = C.int(config.SileroVad.WindowSize)
	c.silero_vad.max_speech_duration = C.float(config.SileroVad.MaxSpeechDuration)

	c.sample_rate = C.int(config.SampleRate)
	c.num_threads = C.int(config.NumThreads)
	c.provider = C.CString(config.Provider)
	defer C.free(unsafe.Pointer(c.provider))

	c.debug = C.int(config.Debug)

	vad := &VoiceActivityDetector{}
	vad.impl = C.SherpaOnnxCreateVoiceActivityDetector(&c, C.float(bufferSizeInSeconds))

	return vad
}

func DeleteVoiceActivityDetector(vad *VoiceActivityDetector) {
	C.SherpaOnnxDestroyVoiceActivityDetector(vad.impl)
	vad.impl = nil
}

func (vad *VoiceActivityDetector) AcceptWaveform(samples []float32) {
	C.SherpaOnnxVoiceActivityDetectorAcceptWaveform(vad.impl, (*C.float)(&samples[0]), C.int(len(samples)))
}

func (vad *VoiceActivityDetector) IsEmpty() bool {
	return int(C.SherpaOnnxVoiceActivityDetectorEmpty(vad.impl)) == 1
}

func (vad *VoiceActivityDetector) IsSpeech() bool {
	return int(C.SherpaOnnxVoiceActivityDetectorDetected(vad.impl)) == 1
}

func (vad *VoiceActivityDetector) Pop() {
	C.SherpaOnnxVoiceActivityDetectorPop(vad.impl)
}

func (vad *VoiceActivityDetector) Clear() {
	C.SherpaOnnxVoiceActivityDetectorClear(vad.impl)
}

func (vad *VoiceActivityDetector) Front() *SpeechSegment {
	f := C.SherpaOnnxVoiceActivityDetectorFront(vad.impl)
	defer C.SherpaOnnxDestroySpeechSegment(f)

	ans := &SpeechSegment{}
	ans.Start = int(f.start)

	n := int(f.n)
	ans.Samples = make([]float32, n)

	samples := (*[1 << 28]C.float)(unsafe.Pointer(f.samples))[:n:n]

	for i := 0; i < n; i++ {
		ans.Samples[i] = float32(samples[i])
	}

	return ans
}

func (vad *VoiceActivityDetector) Reset() {
	C.SherpaOnnxVoiceActivityDetectorReset(vad.impl)
}

func (vad *VoiceActivityDetector) Flush() {
	C.SherpaOnnxVoiceActivityDetectorFlush(vad.impl)
}

// Spoken language identification

type SpokenLanguageIdentificationWhisperConfig struct {
	Encoder      string
	Decoder      string
	TailPaddings int
}

type SpokenLanguageIdentificationConfig struct {
	Whisper    SpokenLanguageIdentificationWhisperConfig
	NumThreads int
	Debug      int
	Provider   string
}

type SpokenLanguageIdentification struct {
	impl *C.struct_SherpaOnnxSpokenLanguageIdentification
}

type SpokenLanguageIdentificationResult struct {
	Lang string
}

func NewSpokenLanguageIdentification(config *SpokenLanguageIdentificationConfig) *SpokenLanguageIdentification {
	c := C.struct_SherpaOnnxSpokenLanguageIdentificationConfig{}

	c.whisper.encoder = C.CString(config.Whisper.Encoder)
	defer C.free(unsafe.Pointer(c.whisper.encoder))

	c.whisper.decoder = C.CString(config.Whisper.Decoder)
	defer C.free(unsafe.Pointer(c.whisper.decoder))

	c.whisper.tail_paddings = C.int(config.Whisper.TailPaddings)

	c.num_threads = C.int(config.NumThreads)
	c.debug = C.int(config.Debug)

	c.provider = C.CString(config.Provider)
	defer C.free(unsafe.Pointer(c.provider))

	slid := &SpokenLanguageIdentification{}
	slid.impl = C.SherpaOnnxCreateSpokenLanguageIdentification(&c)

	return slid
}

func DeleteSpokenLanguageIdentification(slid *SpokenLanguageIdentification) {
	C.SherpaOnnxDestroySpokenLanguageIdentification(slid.impl)
	slid.impl = nil
}

// The user has to invoke DeleteOfflineStream() to free the returned value
// to avoid memory leak
func (slid *SpokenLanguageIdentification) CreateStream() *OfflineStream {
	stream := &OfflineStream{}
	stream.impl = C.SherpaOnnxSpokenLanguageIdentificationCreateOfflineStream(slid.impl)
	return stream
}

func (slid *SpokenLanguageIdentification) Compute(stream *OfflineStream) *SpokenLanguageIdentificationResult {
	r := C.SherpaOnnxSpokenLanguageIdentificationCompute(slid.impl, stream.impl)
	// defer C.SherpaOnnxDestroySpokenLanguageIdentificationResult(r)

	ans := &SpokenLanguageIdentificationResult{}
	ans.Lang = C.GoString(r.lang)

	return ans
}

// ============================================================
// For speaker embedding extraction
// ============================================================

type SpeakerEmbeddingExtractorConfig struct {
	Model      string
	NumThreads int
	Debug      int
	Provider   string
}

type SpeakerEmbeddingExtractor struct {
	impl *C.struct_SherpaOnnxSpeakerEmbeddingExtractor
}

// The user has to invoke [DeleteSpeakerEmbeddingExtractor]() to free the returned value
// to avoid memory leak
func NewSpeakerEmbeddingExtractor(config *SpeakerEmbeddingExtractorConfig) *SpeakerEmbeddingExtractor {
	c := C.struct_SherpaOnnxSpeakerEmbeddingExtractorConfig{}

	c.model = C.CString(config.Model)
	defer C.free(unsafe.Pointer(c.model))

	c.num_threads = C.int(config.NumThreads)
	c.debug = C.int(config.Debug)

	c.provider = C.CString(config.Provider)
	defer C.free(unsafe.Pointer(c.provider))

	ex := &SpeakerEmbeddingExtractor{}
	ex.impl = C.SherpaOnnxCreateSpeakerEmbeddingExtractor(&c)

	return ex
}

func DeleteSpeakerEmbeddingExtractor(ex *SpeakerEmbeddingExtractor) {
	C.SherpaOnnxDestroySpeakerEmbeddingExtractor(ex.impl)
	ex.impl = nil
}

func (ex *SpeakerEmbeddingExtractor) Dim() int {
	return int(C.SherpaOnnxSpeakerEmbeddingExtractorDim(ex.impl))
}

// The user is responsible to invoke [DeleteOnlineStream]() to free
// the returned stream to avoid memory leak
func (ex *SpeakerEmbeddingExtractor) CreateStream() *OnlineStream {
	stream := &OnlineStream{}
	stream.impl = C.SherpaOnnxSpeakerEmbeddingExtractorCreateStream(ex.impl)
	return stream
}

func (ex *SpeakerEmbeddingExtractor) IsReady(stream *OnlineStream) bool {
	return int(C.SherpaOnnxSpeakerEmbeddingExtractorIsReady(ex.impl, stream.impl)) == 1
}

func (ex *SpeakerEmbeddingExtractor) Compute(stream *OnlineStream) []float32 {
	embedding := C.SherpaOnnxSpeakerEmbeddingExtractorComputeEmbedding(ex.impl, stream.impl)
	defer C.SherpaOnnxSpeakerEmbeddingExtractorDestroyEmbedding(embedding)

	n := ex.Dim()
	ans := make([]float32, n)

	// see https://stackoverflow.com/questions/48756732/what-does-1-30c-yourtype-do-exactly-in-cgo
	// :n:n means 0:n:n, means low:high:capacity
	c := (*[1 << 28]C.float)(unsafe.Pointer(embedding))[:n:n]

	for i := 0; i < n; i++ {
		ans[i] = float32(c[i])
	}

	return ans
}

type SpeakerEmbeddingManager struct {
	impl *C.struct_SherpaOnnxSpeakerEmbeddingManager
}

// The user has to invoke [DeleteSpeakerEmbeddingManager]() to free the returned
// value to avoid memory leak
func NewSpeakerEmbeddingManager(dim int) *SpeakerEmbeddingManager {
	m := &SpeakerEmbeddingManager{}
	m.impl = C.SherpaOnnxCreateSpeakerEmbeddingManager(C.int(dim))
	return m
}

func DeleteSpeakerEmbeddingManager(m *SpeakerEmbeddingManager) {
	C.SherpaOnnxDestroySpeakerEmbeddingManager(m.impl)
	m.impl = nil
}

func (m *SpeakerEmbeddingManager) Register(name string, embedding []float32) bool {
	s := C.CString(name)
	defer C.free(unsafe.Pointer(s))

	return C.int(C.SherpaOnnxSpeakerEmbeddingManagerAdd(m.impl, s, (*C.float)(&embedding[0]))) == 1
}

func (m *SpeakerEmbeddingManager) RegisterV(name string, embeddings [][]float32) bool {
	s := C.CString(name)
	defer C.free(unsafe.Pointer(s))

	if len(embeddings) == 0 {
		return false
	}

	dim := len(embeddings[0])
	v := make([]float32, 0, dim*len(embeddings))
	for _, embedding := range embeddings {
		v = append(v, embedding...)
	}

	return C.int(C.SherpaOnnxSpeakerEmbeddingManagerAddListFlattened(m.impl, s, (*C.float)(&v[0]), C.int(len(embeddings)))) == 1
}

func (m *SpeakerEmbeddingManager) Remove(name string) bool {
	s := C.CString(name)
	defer C.free(unsafe.Pointer(s))

	return C.int(C.SherpaOnnxSpeakerEmbeddingManagerRemove(m.impl, s)) == 1
}

func (m *SpeakerEmbeddingManager) Search(embedding []float32, threshold float32) string {
	var s string

	name := C.SherpaOnnxSpeakerEmbeddingManagerSearch(m.impl, (*C.float)(&embedding[0]), C.float(threshold))
	defer C.SherpaOnnxSpeakerEmbeddingManagerFreeSearch(name)

	if name != nil {
		s = C.GoString(name)
	}

	return s
}

func (m *SpeakerEmbeddingManager) Verify(name string, embedding []float32, threshold float32) bool {
	s := C.CString(name)
	defer C.free(unsafe.Pointer(s))

	return C.int(C.SherpaOnnxSpeakerEmbeddingManagerVerify(m.impl, s, (*C.float)(&embedding[0]), C.float(threshold))) == 1
}

func (m *SpeakerEmbeddingManager) Contains(name string) bool {
	s := C.CString(name)
	defer C.free(unsafe.Pointer(s))

	return C.int(C.SherpaOnnxSpeakerEmbeddingManagerContains(m.impl, s)) == 1
}

func (m *SpeakerEmbeddingManager) NumSpeakers() int {
	return int(C.SherpaOnnxSpeakerEmbeddingManagerNumSpeakers(m.impl))
}

func (m *SpeakerEmbeddingManager) AllSpeakers() []string {
	all_speakers := C.SherpaOnnxSpeakerEmbeddingManagerGetAllSpeakers(m.impl)
	defer C.SherpaOnnxSpeakerEmbeddingManagerFreeAllSpeakers(all_speakers)

	n := m.NumSpeakers()
	if n == 0 {
		return nil
	}

	// https://stackoverflow.com/questions/62012070/convert-array-of-strings-from-cgo-in-go
	p := (*[1 << 28]*C.char)(unsafe.Pointer(all_speakers))[:n:n]

	ans := make([]string, n)

	for i := 0; i < n; i++ {
		ans[i] = C.GoString(p[i])
	}

	return ans
}

// Wave

// single channel wave
type Wave = GeneratedAudio

func ReadWave(filename string) *Wave {
	s := C.CString(filename)
	defer C.free(unsafe.Pointer(s))

	w := C.SherpaOnnxReadWave(s)
	defer C.SherpaOnnxFreeWave(w)

	if w == nil {
		return nil
	}

	n := int(w.num_samples)
	if n == 0 {
		return nil
	}

	ans := &Wave{}
	ans.SampleRate = int(w.sample_rate)
	samples := (*[1 << 28]C.float)(unsafe.Pointer(w.samples))[:n:n]

	ans.Samples = make([]float32, n)

	for i := 0; i < n; i++ {
		ans.Samples[i] = float32(samples[i])
	}

	return ans
}

// ============================================================
// For offline speaker diarization
// ============================================================
type OfflineSpeakerSegmentationPyannoteModelConfig struct {
	Model string
}

type OfflineSpeakerSegmentationModelConfig struct {
	Pyannote   OfflineSpeakerSegmentationPyannoteModelConfig
	NumThreads int
	Debug      int
	Provider   string
}

type FastClusteringConfig struct {
	NumClusters int
	Threshold   float32
}

type OfflineSpeakerDiarizationConfig struct {
	Segmentation   OfflineSpeakerSegmentationModelConfig
	Embedding      SpeakerEmbeddingExtractorConfig
	Clustering     FastClusteringConfig
	MinDurationOn  float32
	MinDurationOff float32
}

type OfflineSpeakerDiarization struct {
	impl *C.struct_SherpaOnnxOfflineSpeakerDiarization
}

func DeleteOfflineSpeakerDiarization(sd *OfflineSpeakerDiarization) {
	C.SherpaOnnxDestroyOfflineSpeakerDiarization(sd.impl)
	sd.impl = nil
}

func NewOfflineSpeakerDiarization(config *OfflineSpeakerDiarizationConfig) *OfflineSpeakerDiarization {
	c := C.struct_SherpaOnnxOfflineSpeakerDiarizationConfig{}
	c.segmentation.pyannote.model = C.CString(config.Segmentation.Pyannote.Model)
	defer C.free(unsafe.Pointer(c.segmentation.pyannote.model))

	c.segmentation.num_threads = C.int(config.Segmentation.NumThreads)

	c.segmentation.debug = C.int(config.Segmentation.Debug)

	c.segmentation.provider = C.CString(config.Segmentation.Provider)
	defer C.free(unsafe.Pointer(c.segmentation.provider))

	c.embedding.model = C.CString(config.Embedding.Model)
	defer C.free(unsafe.Pointer(c.embedding.model))

	c.embedding.num_threads = C.int(config.Embedding.NumThreads)

	c.embedding.debug = C.int(config.Embedding.Debug)

	c.embedding.provider = C.CString(config.Embedding.Provider)
	defer C.free(unsafe.Pointer(c.embedding.provider))

	c.clustering.num_clusters = C.int(config.Clustering.NumClusters)
	c.clustering.threshold = C.float(config.Clustering.Threshold)
	c.min_duration_on = C.float(config.MinDurationOn)
	c.min_duration_off = C.float(config.MinDurationOff)

	p := C.SherpaOnnxCreateOfflineSpeakerDiarization(&c)

	if p == nil {
		return nil
	}

	sd := &OfflineSpeakerDiarization{}
	sd.impl = p

	return sd
}

func (sd *OfflineSpeakerDiarization) SampleRate() int {
	return int(C.SherpaOnnxOfflineSpeakerDiarizationGetSampleRate(sd.impl))
}

type OfflineSpeakerDiarizationSegment struct {
	Start   float32
	End     float32
	Speaker int
}

func (sd *OfflineSpeakerDiarization) Process(samples []float32) []OfflineSpeakerDiarizationSegment {
	r := C.SherpaOnnxOfflineSpeakerDiarizationProcess(sd.impl, (*C.float)(&samples[0]), C.int(len(samples)))
	defer C.SherpaOnnxOfflineSpeakerDiarizationDestroyResult(r)

	n := int(C.SherpaOnnxOfflineSpeakerDiarizationResultGetNumSegments(r))

	if n == 0 {
		return nil
	}

	s := C.SherpaOnnxOfflineSpeakerDiarizationResultSortByStartTime(r)
	defer C.SherpaOnnxOfflineSpeakerDiarizationDestroySegment(s)

	ans := make([]OfflineSpeakerDiarizationSegment, n)

	p := (*[1 << 28]C.struct_SherpaOnnxOfflineSpeakerDiarizationSegment)(unsafe.Pointer(s))[:n:n]

	for i := 0; i < n; i++ {
		ans[i].Start = float32(p[i].start)
		ans[i].End = float32(p[i].end)
		ans[i].Speaker = int(p[i].speaker)
	}

	return ans
}
