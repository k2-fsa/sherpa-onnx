// kws_api.h - KWS 关键词检测对外 API（不依赖 sherpa-onnx 头文件）
//
// 使用方式：仅需包含本头文件并链接 libkws_sdk，运行时需同时链接 sherpa-onnx 库。
#ifndef KWS_API_H_
#define KWS_API_H_

#include <string>
#include <vector>

namespace kws {

/// 配置：模型路径、关键词文件、采样率等
struct Config {
  std::string encoder_path;
  std::string decoder_path;
  std::string joiner_path;
  std::string tokens_path;
  std::string keywords_file_path;
  int sample_rate = 16000;
  int num_threads = 1;
  float keywords_score = 2.0f;
  float keywords_threshold = 0.15f;
  int num_trailing_blanks = 1;
  int max_active_paths = 4;
};

/// 单次检测结果
struct Result {
  /// 命中的关键词（空串表示未命中或需用 full_tokens 做整句过滤）
  std::string keyword;
  /// 本段完整解码 token 序列，用于整句匹配：仅当与某关键词 token 序列完全一致时算命中
  std::vector<std::string> full_tokens;
  std::vector<float> timestamps;
  float start_time = 0.f;
};

/// 引擎句柄（不透明，仅通过本 API 使用）
struct Engine;
/// 流句柄（不透明）
struct Stream;

/// 创建引擎，失败返回 nullptr
Engine* Create(const Config& config);
void Destroy(Engine* engine);

/// 创建/销毁流（每路会话一条流）
Stream* CreateStream(Engine* engine);
void DestroyStream(Stream* stream);

/// 送入一帧 PCM：float、单声道，采样率与 config.sample_rate 一致
void AcceptWaveform(Engine* engine, Stream* stream, int sample_rate,
                    const float* data, int num_samples);

/// 是否有待解码数据
bool IsReady(Engine* engine, Stream* stream);
/// 解码
void Decode(Engine* engine, Stream* stream);
/// 取结果（命中后需调用 Reset）
Result GetResult(Engine* engine, Stream* stream);
/// 命中后必须调用，重置流状态
void Reset(Engine* engine, Stream* stream);

/// 整句匹配辅助：若 full_tokens 与 keyword_tokens 完全一致返回 true
inline bool IsExactMatch(const Result& r,
                         const std::vector<std::string>& keyword_tokens) {
  if (r.full_tokens.size() != keyword_tokens.size()) return false;
  for (size_t i = 0; i < r.full_tokens.size(); ++i)
    if (r.full_tokens[i] != keyword_tokens[i]) return false;
  return true;
}

}  // namespace kws

#endif  // KWS_API_H_
