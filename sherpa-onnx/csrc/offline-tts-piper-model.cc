// sherpa-onnx/csrc/offline-tts-piper-model.cc
//
// Copyright (c)  2025  Xiaomi Corporation

#include "sherpa-onnx/csrc/offline-tts-piper-model.h"

#include <algorithm>
#include <fstream>
#include <sstream>
#include <utility>

#if __ANDROID_API__ >= 9
#include "android/asset_manager.h"
#include "android/asset_manager_jni.h"
#endif

#include "sherpa-onnx/csrc/file-utils.h"
#include "sherpa-onnx/csrc/macros.h"
#include "sherpa-onnx/csrc/onnx-utils.h"
#include "sherpa-onnx/csrc/session.h"
#include "sherpa-onnx/csrc/text-utils.h"

namespace sherpa_onnx {

class OfflineTtsPiperModel::Impl {
 public:
  Impl(const OfflineTtsModelConfig &config)
      : config_(config),
        env_(ORT_LOGGING_LEVEL_WARNING),
        sess_opts_(GetSessionOptions(config)),
        allocator_{} {
    {
      auto buf = ReadFile(config.piper.model);
      InitModel(buf);
    }

    // Initialize with default values or try to read from ONNX model metadata
    InitFromOnnxModel();
  }

#if __ANDROID_API__ >= 9
  Impl(AAssetManager *mgr, const OfflineTtsModelConfig &config)
      : config_(config),
        env_(ORT_LOGGING_LEVEL_WARNING),
        sess_opts_(GetSessionOptions(config)),
        allocator_{} {
    {
      auto buf = ReadFile(mgr, config.piper.model);
      InitModel(buf);
    }

    // Initialize with default values or try to read from ONNX model metadata
    InitFromOnnxModel();
  }
#endif

  Ort::Value Run(Ort::Value phoneme_ids, int64_t speaker_id, float speed) const {
    // For now, just return a dummy tensor
    // TODO: Implement actual Piper model inference
    
    std::array<int64_t, 2> output_shape{1, 22050}; // 1 second at 22050 Hz
    
    auto output = Ort::Value::CreateTensor<float>(
        allocator_, output_shape.data(), output_shape.size());
    
    // Fill with zeros for now
    float *output_data = output.GetTensorMutableData<float>();
    std::fill_n(output_data, 22050, 0.0f);
    
    return output;
  }

  OrtAllocator *Allocator() const { return allocator_; }

  const OfflineTtsPiperModelMetaData &GetMetaData() const { return meta_data_; }

 private:
  void InitModel(const std::vector<char> &model_data) {
    sess_ = std::make_unique<Ort::Session>(env_, model_data.data(),
                                           model_data.size(), sess_opts_);

    GetInputNames(sess_.get(), &input_names_, &input_names_ptr_);
    GetOutputNames(sess_.get(), &output_names_, &output_names_ptr_);
  }

  void InitFromOnnxModel() {
    // First try to read from JSON config file if provided
    bool config_loaded = false;
    if (!config_.piper.model_config_file.empty()) {
      config_loaded = LoadConfigFromJson();
    }
    
    if (!config_loaded) {
      // Try to read configuration from ONNX model metadata as fallback
      try {
        Ort::ModelMetadata meta_data = sess_->GetModelMetadata();
        Ort::AllocatorWithDefaultOptions allocator;
        
        // Try to read sample rate from model metadata
        SHERPA_ONNX_READ_META_DATA_WITH_DEFAULT(meta_data_.sample_rate, "sample_rate", 22050);
        SHERPA_ONNX_READ_META_DATA_WITH_DEFAULT(meta_data_.num_speakers, "num_speakers", 1);
        
        if (config_.debug) {
          std::ostringstream os;
          os << "---piper model---\n";
          PrintModelMetadata(os, meta_data);
          
          os << "----------input names----------\n";
          int32_t i = 0;
          for (const auto &s : input_names_) {
            os << i << " " << s << "\n";
            ++i;
          }
          os << "----------output names----------\n";
          i = 0;
          for (const auto &s : output_names_) {
            os << i << " " << s << "\n";
            ++i;
          }
          
          SHERPA_ONNX_LOGE("%s\n", os.str().c_str());
        }
      } catch (...) {
        // If reading from metadata fails, use defaults
        SHERPA_ONNX_LOGE("Failed to read Piper model metadata, using defaults");
      }
      
      // Initialize default phoneme ID mappings as fallback
      InitDefaultPhonemeIdMap();
    }
  }

  // Simple JSON config parser for Piper models
  bool LoadConfigFromJson() {
    std::ifstream config_file(config_.piper.model_config_file);
    if (!config_file.is_open()) {
      SHERPA_ONNX_LOGE("Failed to open Piper config file: %s", 
                       config_.piper.model_config_file.c_str());
      return false;
    }
    
    // Read entire file
    std::string config_content;
    std::string line;
    while (std::getline(config_file, line)) {
      config_content += line;
    }
    config_file.close();
    
    // Parse basic JSON fields manually (simple implementation)
    return ParsePiperConfig(config_content);
  }
  
  bool ParsePiperConfig(const std::string& json_content) {
    // Simple JSON parsing for essential Piper config fields
    // This is a basic implementation - a real JSON parser would be better
    
    try {
      // Parse audio section
      auto audio_pos = json_content.find("\"audio\"");
      if (audio_pos != std::string::npos) {
        auto sample_rate_pos = json_content.find("\"sample_rate\"", audio_pos);
        if (sample_rate_pos != std::string::npos) {
          auto colon_pos = json_content.find(":", sample_rate_pos);
          if (colon_pos != std::string::npos) {
            auto value_start = json_content.find_first_of("0123456789", colon_pos);
            if (value_start != std::string::npos) {
              auto value_end = json_content.find_first_not_of("0123456789", value_start);
              std::string value_str = json_content.substr(value_start, value_end - value_start);
              meta_data_.sample_rate = std::stoi(value_str);
            }
          }
        }
      }
      
      // Parse num_speakers
      auto speakers_pos = json_content.find("\"num_speakers\"");
      if (speakers_pos != std::string::npos) {
        auto colon_pos = json_content.find(":", speakers_pos);
        if (colon_pos != std::string::npos) {
          auto value_start = json_content.find_first_of("0123456789", colon_pos);
          if (value_start != std::string::npos) {
            auto value_end = json_content.find_first_not_of("0123456789", value_start);
            std::string value_str = json_content.substr(value_start, value_end - value_start);
            meta_data_.num_speakers = std::stoi(value_str);
          }
        }
      }
      
      // Parse phoneme_id_map
      auto phoneme_map_pos = json_content.find("\"phoneme_id_map\"");
      if (phoneme_map_pos != std::string::npos) {
        ParsePhonemeIdMap(json_content, phoneme_map_pos);
      } else {
        // No phoneme_id_map found, use defaults
        InitDefaultPhonemeIdMap();
      }
      
      if (config_.debug) {
        SHERPA_ONNX_LOGE("Loaded Piper config: sample_rate=%d, num_speakers=%d, phoneme_map_size=%zu",
                         meta_data_.sample_rate, meta_data_.num_speakers, 
                         meta_data_.phoneme_id_map.size());
      }
      
      return true;
    } catch (const std::exception& e) {
      SHERPA_ONNX_LOGE("Error parsing Piper config: %s", e.what());
      return false;
    }
  }
  
  void ParsePhonemeIdMap(const std::string& json_content, size_t start_pos) {
    // Find the opening brace of phoneme_id_map
    auto brace_start = json_content.find("{", start_pos);
    if (brace_start == std::string::npos) return;
    
    // Find matching closing brace
    int brace_count = 1;
    size_t pos = brace_start + 1;
    size_t brace_end = std::string::npos;
    
    while (pos < json_content.size() && brace_count > 0) {
      if (json_content[pos] == '{') brace_count++;
      else if (json_content[pos] == '}') {
        brace_count--;
        if (brace_count == 0) {
          brace_end = pos;
          break;
        }
      }
      pos++;
    }
    
    if (brace_end == std::string::npos) return;
    
    // Extract the phoneme map content
    std::string map_content = json_content.substr(brace_start + 1, brace_end - brace_start - 1);
    
    // Parse individual phoneme mappings
    // Format: "phoneme": [id1, id2, ...]
    size_t search_pos = 0;
    while (true) {
      auto quote_start = map_content.find("\"", search_pos);
      if (quote_start == std::string::npos) break;
      
      auto quote_end = map_content.find("\"", quote_start + 1);
      if (quote_end == std::string::npos) break;
      
      std::string phoneme_str = map_content.substr(quote_start + 1, quote_end - quote_start - 1);
      
      // Find the array of IDs
      auto colon_pos = map_content.find(":", quote_end);
      if (colon_pos == std::string::npos) break;
      
      auto bracket_start = map_content.find("[", colon_pos);
      if (bracket_start == std::string::npos) break;
      
      auto bracket_end = map_content.find("]", bracket_start);
      if (bracket_end == std::string::npos) break;
      
      // Parse the first ID (assuming single ID per phoneme for simplicity)
      std::string ids_str = map_content.substr(bracket_start + 1, bracket_end - bracket_start - 1);
      auto comma_pos = ids_str.find(",");
      if (comma_pos != std::string::npos) {
        ids_str = ids_str.substr(0, comma_pos);
      }
      
      // Remove whitespace
      ids_str.erase(std::remove_if(ids_str.begin(), ids_str.end(), ::isspace), ids_str.end());
      
      if (!ids_str.empty() && std::all_of(ids_str.begin(), ids_str.end(), ::isdigit)) {
        int64_t phoneme_id = std::stoll(ids_str);
        
        // Convert phoneme string to char32_t (UTF-32)
        if (phoneme_str.size() == 1) {
          // ASCII character
          char32_t phoneme_char = static_cast<char32_t>(phoneme_str[0]);
          meta_data_.phoneme_id_map[phoneme_char] = phoneme_id;
        } else if (phoneme_str.size() > 1) {
          // Assume UTF-8, convert to first codepoint
          // Simple implementation for common cases
          if (phoneme_str == "_") meta_data_.phoneme_id_map[U'_'] = phoneme_id;
          else if (phoneme_str == "^") meta_data_.phoneme_id_map[U'^'] = phoneme_id;
          else if (phoneme_str == "$") meta_data_.phoneme_id_map[U'$'] = phoneme_id;
          // Add more special cases as needed
        }
      }
      
      search_pos = bracket_end + 1;
    }
    
    if (config_.debug) {
      SHERPA_ONNX_LOGE("Parsed %zu phoneme mappings from config", meta_data_.phoneme_id_map.size());
    }
  }

  void InitDefaultPhonemeIdMap() {
    // Initialize with basic IPA phoneme mappings for Piper
    meta_data_.phoneme_id_map[U'_'] = 0;   // pad
    meta_data_.phoneme_id_map[U'^'] = 1;   // bos  
    meta_data_.phoneme_id_map[U'$'] = 2;   // eos
    meta_data_.phoneme_id_map[U' '] = 3;   // space
    
    // Add basic English phonemes (matching Piper's default mapping)
    // These are just examples - real Piper models may have different mappings
    char32_t phonemes[] = {
      U'!', U'\'', U'(', U')', U',', U'-', U'.', U':', U';', U'?',
      U'a', U'b', U'c', U'd', U'e', U'f', U'h', U'i', U'j', U'k',
      U'l', U'm', U'n', U'o', U'p', U'q', U'r', U's', U't', U'u',
      U'v', U'w', U'x', U'y', U'z'
    };
    
    int64_t id = 4; // Start after pad, bos, eos, space
    for (auto phoneme : phonemes) {
      meta_data_.phoneme_id_map[phoneme] = id++;
    }
  }

  OfflineTtsModelConfig config_;
  OfflineTtsPiperModelMetaData meta_data_;

  Ort::Env env_;
  Ort::SessionOptions sess_opts_;
  Ort::AllocatorWithDefaultOptions allocator_;

  std::unique_ptr<Ort::Session> sess_;

  std::vector<std::string> input_names_;
  std::vector<const char *> input_names_ptr_;

  std::vector<std::string> output_names_;
  std::vector<const char *> output_names_ptr_;
};

OfflineTtsPiperModel::OfflineTtsPiperModel(const OfflineTtsModelConfig &config)
    : impl_(std::make_unique<Impl>(config)) {}

#if __ANDROID_API__ >= 9
OfflineTtsPiperModel::OfflineTtsPiperModel(AAssetManager *mgr,
                                           const OfflineTtsModelConfig &config)
    : impl_(std::make_unique<Impl>(mgr, config)) {}
#endif

OfflineTtsPiperModel::~OfflineTtsPiperModel() = default;

Ort::Value OfflineTtsPiperModel::Run(Ort::Value phoneme_ids, int64_t speaker_id,
                                     float speed) const {
  return impl_->Run(std::move(phoneme_ids), speaker_id, speed);
}

OrtAllocator *OfflineTtsPiperModel::Allocator() const {
  return impl_->Allocator();
}

const OfflineTtsPiperModelMetaData &OfflineTtsPiperModel::GetMetaData() const {
  return impl_->GetMetaData();
}

}  // namespace sherpa_onnx