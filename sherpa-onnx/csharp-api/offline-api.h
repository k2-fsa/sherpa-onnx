// sherpa-onnx/sharp-api/offline-api.h
//
// Copyright (c)  2023  Manyeyes Corporation

#pragma once

#include <list>

namespace sherpa_onnx
{
	/// Please refer to
	/// https://k2-fsa.github.io/sherpa/onnx/pretrained_models/index.html
	/// to download pre-trained models. That is, you can find encoder-xxx.onnx
	/// decoder-xxx.onnx, joiner-xxx.onnx, and tokens.txt for this struct
	/// from there.
	typedef struct SherpaOnnxOfflineTransducer {
		const char* encoder_filename;
		const char* decoder_filename;
		const char* joiner_filename;
	} SherpaOnnxOfflineTransducer;

	typedef struct SherpaOnnxOfflineParaformer {
		const char* model;
	}SherpaOnnxOfflineParaformer;

	typedef struct SherpaOnnxOfflineNemoEncDecCtc {
		const char* model;
	}SherpaOnnxOfflineNemoEncDecCtc;


	typedef struct SherpaOnnxOfflineModelConfig {
		SherpaOnnxOfflineTransducer transducer;
		SherpaOnnxOfflineParaformer paraformer;
		SherpaOnnxOfflineNemoEncDecCtc nemo_ctc;
		const char* tokens;
		const int32_t num_threads;
		const bool debug;
	} SherpaOnnxOfflineModelConfig;

	/// It expects 16 kHz 16-bit single channel wave format.
	typedef struct SherpaOnnxFeatureConfig {
		/// Sample rate of the input data. MUST match the one expected
		/// by the model. For instance, it should be 16000 for models provided
		/// by us.
		int32_t sample_rate;

		/// Feature dimension of the model.
		/// For instance, it should be 80 for models provided by us.
		int32_t feature_dim;
	} SherpaOnnxFeatureConfig;

	typedef struct SherpaOnnxOfflineRecognizerConfig {
		SherpaOnnxFeatureConfig feat_config;
		SherpaOnnxOfflineModelConfig model_config;

		/// Possible values are: greedy_search, modified_beam_search
		const char* decoding_method;

	} SherpaOnnxOfflineRecognizerConfig;

	typedef struct SherpaOnnxOfflineRecognizerResult {
		// Recognition results.
		// For English, it consists of space separated words.
		// For Chinese, it consists of Chinese words without spaces.
		char* text;
		int text_len;

		// Decoded results at the token level.
		// For instance, for BPE-based models it consists of a list of BPE tokens.
		// std::vector<std::string> tokens;

		// timestamps.size() == tokens.size()
		// timestamps[i] records the time in seconds when tokens[i] is decoded.
		// std::vector<float> timestamps;
	} SherpaOnnxOfflineRecognizerResult;

	/// Note: OfflineRecognizer here means StreamingRecognizer.
	/// It does not need to access the Internet during recognition.
	/// Everything is run locally.
	typedef struct SherpaOnnxOfflineRecognizer SherpaOnnxOfflineRecognizer;

	typedef struct SherpaOnnxOfflineStream SherpaOnnxOfflineStream;

	extern "C" __declspec(dllexport)
		SherpaOnnxOfflineRecognizer * __stdcall  CreateOfflineRecognizer(
			const SherpaOnnxOfflineRecognizerConfig * config);

	extern "C" __declspec(dllexport)
		SherpaOnnxOfflineStream * __stdcall CreateOfflineStream(
			SherpaOnnxOfflineRecognizer * sherpaOnnxOfflineRecognizer);

	extern "C" __declspec(dllexport)
		void __stdcall AcceptWaveform(
			SherpaOnnxOfflineStream * stream, int32_t sample_rate,
			const float* samples, int32_t samples_size);

	extern "C" __declspec(dllexport)
		void __stdcall DecodeOfflineStream(
			SherpaOnnxOfflineRecognizer * recognizer,
			SherpaOnnxOfflineStream * stream);

	extern "C" __declspec(dllexport)
		void __stdcall DecodeMultipleOfflineStreams(
			SherpaOnnxOfflineRecognizer * recognizer,
			SherpaOnnxOfflineStream * *streams, int32_t n);

	extern "C" __declspec(dllexport)
		SherpaOnnxOfflineRecognizerResult * __stdcall GetOfflineStreamResult(
			SherpaOnnxOfflineStream * stream);

	extern "C" __declspec(dllexport)
		void __stdcall DestroyOfflineRecognizer(
			SherpaOnnxOfflineRecognizer * recognizer);

	extern "C" __declspec(dllexport)
		void __stdcall DestroyOfflineStream(
			SherpaOnnxOfflineStream * stream);

	extern "C" __declspec(dllexport)
		void __stdcall DestroyOfflineRecognizerResult(
			SherpaOnnxOfflineRecognizerResult * r);
}// namespace sherpa_onnx