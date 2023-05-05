// sherpa-onnx/sharp-api/offline-api.cpp
//
// Copyright (c)  2023  Manyeyes Corporation

#include "offline-api.h"

#include "sherpa-onnx/csrc/display.h"
#include "sherpa-onnx/csrc/offline-recognizer.h"

namespace sherpa_onnx
{
	struct SherpaOnnxOfflineRecognizer {
		sherpa_onnx::OfflineRecognizer* impl;
	};

	struct SherpaOnnxOfflineStream {
		std::unique_ptr<sherpa_onnx::OfflineStream> impl;
		explicit SherpaOnnxOfflineStream(std::unique_ptr<sherpa_onnx::OfflineStream> p)
			: impl(std::move(p)) {}
	};

	struct SherpaOnnxDisplay {
		std::unique_ptr<sherpa_onnx::Display> impl;
	};

	SherpaOnnxOfflineRecognizer* __stdcall CreateOfflineRecognizer(
	    const SherpaOnnxOfflineRecognizerConfig* config) {
		sherpa_onnx::OfflineRecognizerConfig recognizer_config;

		recognizer_config.feat_config.sampling_rate = config->feat_config.sample_rate;
		recognizer_config.feat_config.feature_dim = config->feat_config.feature_dim;

		if (strlen(config->model_config.transducer.encoder_filename) > 0) {
			recognizer_config.model_config.transducer.encoder_filename =
				config->model_config.transducer.encoder_filename;
			recognizer_config.model_config.transducer.decoder_filename =
				config->model_config.transducer.decoder_filename;
			recognizer_config.model_config.transducer.joiner_filename =
				config->model_config.transducer.joiner_filename;
		}
		else if (strlen(config->model_config.paraformer.model) > 0) {
			recognizer_config.model_config.paraformer.model =
				config->model_config.paraformer.model;
		}
		else if (strlen(config->model_config.nemo_ctc.model) > 0) {
			recognizer_config.model_config.nemo_ctc.model =
				config->model_config.nemo_ctc.model;
		}

		recognizer_config.model_config.tokens =
		    config->model_config.tokens;
		recognizer_config.model_config.num_threads =
		    config->model_config.num_threads;
		recognizer_config.model_config.debug =
		    config->model_config.debug;

		recognizer_config.decoding_method = config->decoding_method;

		SherpaOnnxOfflineRecognizer* recognizer =
		    new SherpaOnnxOfflineRecognizer;
		recognizer->impl =
		    new sherpa_onnx::OfflineRecognizer(recognizer_config);

		return recognizer;
	}

	SherpaOnnxOfflineStream* __stdcall CreateOfflineStream(
	    SherpaOnnxOfflineRecognizer* recognizer) {
		SherpaOnnxOfflineStream* stream =
		    new SherpaOnnxOfflineStream(recognizer->impl->CreateStream());
		return stream;
	}

	void __stdcall AcceptWaveform(
	    SherpaOnnxOfflineStream* stream,
	    int32_t sample_rate,
		const float* samples, int32_t samples_size) {
		std::vector<float> waveform{ samples, samples + samples_size };
		stream->impl->AcceptWaveform(sample_rate, waveform.data(), waveform.size());
	}

	void __stdcall DecodeOfflineStream(
	    SherpaOnnxOfflineRecognizer* recognizer,
		SherpaOnnxOfflineStream* stream) {
		recognizer->impl->DecodeStream(stream->impl.get());
	}

	void __stdcall DecodeMultipleOfflineStreams(
	    SherpaOnnxOfflineRecognizer* recognizer,
		SherpaOnnxOfflineStream** streams, int32_t n) {
		std::vector<sherpa_onnx::OfflineStream*> ss(n);
		for (int32_t i = 0; i != n; ++i) {
			ss[i] = streams[i]->impl.get();
		}
		recognizer->impl->DecodeStreams(ss.data(), n);
	}

	SherpaOnnxOfflineRecognizerResult* __stdcall GetOfflineStreamResult(
	    SherpaOnnxOfflineStream* stream) {
		sherpa_onnx::OfflineRecognitionResult result =
		    stream->impl->GetResult();
		const auto& text = result.text;
		auto r = new SherpaOnnxOfflineRecognizerResult;
		r->text = new char[text.size() + 1];
		std::copy(text.begin(), text.end(), const_cast<char*>(r->text));
		const_cast<char*>(r->text)[text.size()] = 0;
		r->text_len = text.size();
		return r;
	}


	/// Free a pointer returned by CreateOfflineRecognizer()
	///
	/// @param p A pointer returned by CreateOfflineRecognizer()
	void __stdcall DestroyOfflineRecognizer(
	    SherpaOnnxOfflineRecognizer* recognizer) {
		delete recognizer->impl;
		delete recognizer;
	}

	/// Destory an offline stream.
	///
	/// @param stream A pointer returned by CreateOfflineStream()
	void __stdcall DestroyOfflineStream(SherpaOnnxOfflineStream* stream) {
		delete stream;
	}

	/// Destroy the pointer returned by GetOfflineStreamResult().
	///
	/// @param r A pointer returned by GetOfflineStreamResult()
	void __stdcall DestroyOfflineRecognizerResult(
	    SherpaOnnxOfflineRecognizerResult* r) {
		delete r->text;
		delete r;
	}
}// namespace sherpa_onnx