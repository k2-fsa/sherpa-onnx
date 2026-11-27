// sherpa-onnx/csrc/gigaam-coreml-ctc.mm
//
// Copyright (c)  2026  sherpa-onnx contributors

#include "sherpa-onnx/csrc/gigaam-coreml-ctc.h"

#import <CoreML/CoreML.h>
#import <Foundation/Foundation.h>

#include <array>
#include <cstring>
#include <stdexcept>

namespace sherpa_onnx {

static std::runtime_error CoremlError(const char *operation, NSError *error) {
  const char *reason = error.localizedDescription.UTF8String;
  return std::runtime_error(std::string(operation) + ": " +
                            (reason ? reason : "unknown Core ML error"));
}

class GigaAmCoremlCtc::Impl {
 public:
  Impl(const std::string &path, const std::string &provider) {
    if (provider != "coreml" && provider != "cpu") {
      throw std::invalid_argument(
          "Compiled GigaAM Core ML accepts provider=coreml or provider=cpu");
    }

    @autoreleasepool {
      if (@available(macOS 15.0, *)) {
        NSURL *url = [NSURL fileURLWithPath:@(path.c_str()) isDirectory:YES];
        MLModelConfiguration *config = [[MLModelConfiguration alloc] init];
        config.computeUnits =
            provider == "coreml" ? MLComputeUnitsCPUAndGPU : MLComputeUnitsCPUOnly;
        NSError *error = nil;
        model_ = [MLModel modelWithContentsOfURL:url configuration:config error:&error];
        if (!model_) {
          throw CoremlError("Load compiled GigaAM CTC", error);
        }
      } else {
        throw std::runtime_error("Compiled GigaAM Core ML requires macOS 15");
      }

      MLModelDescription *desc = model_.modelDescription;
      NSDictionary *metadata = desc.metadata[MLModelCreatorDefinedKey];
      MLMultiArrayConstraint *audio =
          desc.inputDescriptionsByName[@"features"].multiArrayConstraint;
      MLMultiArrayConstraint *length =
          desc.inputDescriptionsByName[@"feature_lengths"].multiArrayConstraint;
      if (![metadata[@"gigaam_model_type"] isEqualToString:@"v4-ctc-e2e"] ||
          audio.dataType != MLMultiArrayDataTypeFloat32 || audio.shape.count != 3 ||
          audio.shape[0].longLongValue != 1 || audio.shape[1].longLongValue != kChannels ||
          audio.shape[2].longLongValue <= 0 || audio.shape[2].longLongValue > 3000 ||
          length.dataType != MLMultiArrayDataTypeInt32 || ![length.shape isEqualToArray:@[ @1 ]]) {
        throw std::invalid_argument("Unexpected GigaAM Core ML CTC model signature");
      }
      frames_ = audio.shape[2].longLongValue;
    }
  }

  int64_t Frames() const { return frames_; }

  Ort::Value Forward(const Ort::Value &features, int64_t frames, OrtAllocator *allocator) const {
    @autoreleasepool {
      NSError *error = nil;
      MLMultiArray *audio = [[MLMultiArray alloc] initWithShape:@[ @1, @(kChannels), @(frames_) ]
                                                       dataType:MLMultiArrayDataTypeFloat32
                                                          error:&error];
      if (!audio) {
        throw CoremlError("Allocate GigaAM features", error);
      }
      std::memcpy(audio.dataPointer, features.GetTensorData<float>(),
                  sizeof(float) * kChannels * frames_);
      MLMultiArray *length = [[MLMultiArray alloc] initWithShape:@[ @1 ]
                                                        dataType:MLMultiArrayDataTypeInt32
                                                           error:&error];
      if (!length) {
        throw CoremlError("Allocate GigaAM feature length", error);
      }
      *static_cast<int32_t *>(length.dataPointer) = static_cast<int32_t>(frames);
      MLDictionaryFeatureProvider *inputs =
          [[MLDictionaryFeatureProvider alloc] initWithDictionary:@{
            @"features" : [MLFeatureValue featureValueWithMultiArray:audio],
            @"feature_lengths" : [MLFeatureValue featureValueWithMultiArray:length]
          }
                                                            error:&error];
      if (!inputs) {
        throw CoremlError("Prepare GigaAM CTC inputs", error);
      }
      id<MLFeatureProvider> output = [model_ predictionFromFeatures:inputs error:&error];
      if (!output) {
        throw CoremlError("Predict GigaAM CTC", error);
      }
      return CopyLogProbs([output featureValueForName:@"log_probs"].multiArrayValue, frames,
                          allocator);
    }
  }

 private:
  static Ort::Value CopyLogProbs(MLMultiArray *log_probs, int64_t frames, OrtAllocator *allocator) {
    const int64_t rows = frames / kSubsampling;
    if (log_probs.dataType != MLMultiArrayDataTypeFloat32 || log_probs.shape.count != 3 ||
        log_probs.shape[0].longLongValue != 1 || log_probs.shape[1].longLongValue < rows ||
        log_probs.shape[2].longLongValue != kVocabulary) {
      throw std::runtime_error("Unexpected GigaAM Core ML logits shape or type");
    }
    const std::array<int64_t, 3> shape{1, rows, kVocabulary};
    auto result = Ort::Value::CreateTensor<float>(allocator, shape.data(), shape.size());
    const auto *src = static_cast<const float *>(log_probs.dataPointer);
    auto *dst = result.GetTensorMutableData<float>();
    const auto time_stride = log_probs.strides[1].longLongValue;
    const auto token_stride = log_probs.strides[2].longLongValue;
    for (int64_t t = 0; t < rows; ++t) {
      for (int64_t k = 0; k < kVocabulary; ++k) {
        *dst++ = src[t * time_stride + k * token_stride];
      }
    }
    return result;
  }

  MLModel *model_ = nil;
  int64_t frames_ = 0;
};

GigaAmCoremlCtc::GigaAmCoremlCtc(const std::string &path, const std::string &provider)
    : impl_(std::make_unique<Impl>(path, provider)) {}

GigaAmCoremlCtc::~GigaAmCoremlCtc() = default;

int64_t GigaAmCoremlCtc::Frames() const { return impl_->Frames(); }

Ort::Value GigaAmCoremlCtc::Forward(const Ort::Value &features, int64_t frames,
                                    OrtAllocator *allocator) const {
  return impl_->Forward(features, frames, allocator);
}

}  // namespace sherpa_onnx
