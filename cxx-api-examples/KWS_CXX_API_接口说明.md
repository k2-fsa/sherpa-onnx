# KWS（关键词检测）CXX API 接口说明 — 供系统层封装调用

与 VAD 相同，KWS 使用 **sherpa-onnx 的 CXX API**（`sherpa-onnx/c-api/cxx-api.h`），便于系统层用 C++ 封装成接口。

## 1. 参考示例

- **VAD**（系统层已参考）：[cxx-api-examples/vad-cxx-api.cc](https://github.com/k2-fsa/sherpa-onnx/blob/master/cxx-api-examples/vad-cxx-api.cc)  
- **KWS 文件示例**（推荐先看）：[cxx-api-examples/kws-cxx-api.cc](https://github.com/k2-fsa/sherpa-onnx/blob/master/cxx-api-examples/kws-cxx-api.cc)  
- **KWS 麦克风流式**（与 Python 版逻辑一致）：[sherpa-onnx/csrc/sherpa-onnx-keyword-spotter-microphone.cc](https://github.com/k2-fsa/sherpa-onnx/blob/master/sherpa-onnx/csrc/sherpa-onnx-keyword-spotter-microphone.cc)

## 2. 头文件与命名空间

```cpp
#include "sherpa-onnx/c-api/cxx-api.h"

using namespace sherpa_onnx::cxx;
```

## 3. 接口输入与输出（一览）

| 维度 | 内容 |
|------|------|
| **输入** | **创建时**：`KeywordSpotterConfig`（模型路径、`tokens.txt`、`keywords.txt`、采样率等）。**运行时**：PCM 音频，通过 `stream.AcceptWaveform(sample_rate, data, n)` 送入，要求 **float、单声道、16 kHz**（与 `config.feat_config.sampling_rate` 一致）。 |
| **输出** | 每次 `kws.GetResult(&stream)` 返回一个 **`KeywordResult`**：`keyword`（命中的关键词，空串表示未命中）、`tokens`、**`full_tokens`**（整段 token，用于整句匹配）、`timestamps`、`start_time`、`json`。是否“真正命中”由你在系统层用 `full_tokens` 与关键词 token 表做整句匹配决定。 |

---

## 4. 核心类型与接口

| 类型 / 接口 | 说明 |
|-------------|------|
| `KeywordSpotterConfig` | 配置：模型路径、tokens、keywords 文件、线程数等 |
| `KeywordSpotter::Create(config)` | 创建 KWS 引擎（失败返回空，需检查 config） |
| `kws.CreateStream()` | 创建一条流（每路麦克风/会话一条） |
| `stream.AcceptWaveform(sample_rate, data, n)` | 送入 PCM  float、16kHz、单声道 |
| `kws.IsReady(&stream)` | 当前流是否有待解码数据 |
| `kws.Decode(&stream)` | 解码当前流 |
| `kws.GetResult(&stream)` | 取结果，返回 `KeywordResult` |
| `KeywordResult.keyword` | 命中的关键词字符串（空表示未命中） |
| `KeywordResult.tokens` | 命中段的 token 序列（可选使用） |
| `KeywordResult.full_tokens` | **整段解码的 token 序列**，用于「整句匹配」：仅当 full_tokens 等于某关键词的 token 序列时才算真正命中（见下） |
| `KeywordResult.timestamps` | 各 token 时间戳（可选） |
| `kws.Reset(&stream)` | **命中后必须调用**，重置流状态以便下次检测 |

## 5. 推荐调用流程（与 VAD 风格一致）

```text
1) 配置 KeywordSpotterConfig（tokens、encoder、decoder、joiner、keywords_file 等）
2) spotter = KeywordSpotter::Create(config)，检查 spotter.Get() 非空
3) stream = spotter.CreateStream()
4) 循环（例如从麦克风或上层送帧）：
   a. 收到一帧 PCM → stream.AcceptWaveform(sample_rate, data, n)
   b. while (spotter.IsReady(&stream)) {
        spotter.Decode(&stream);
        auto r = spotter.GetResult(&stream);
        if (!r.keyword.empty()) {
          // 命中：r.keyword 即为关键词，可回调/上报
          // 上报后务必：
          spotter.Reset(&stream);
        }
      }
```

**注意**：每次检测到一次命中后必须调用 `Reset(&stream)`，否则后续行为异常。

## 6. 配置示例（按需改成实际路径）

```cpp
KeywordSpotterConfig config;
config.model_config.transducer.encoder = "/path/to/encoder.onnx";
config.model_config.transducer.decoder = "/path/to/decoder.onnx";
config.model_config.transducer.joiner  = "/path/to/joiner.onnx";
config.model_config.tokens             = "/path/to/tokens.txt";
config.model_config.provider          = "cpu";
config.model_config.num_threads       = 1;
config.keywords_file                  = "/path/to/keywords.txt";
config.keywords_score                 = 2.0f;   // 可选，提高易触发
config.keywords_threshold             = 0.15f;  // 可选，触发阈值
config.num_trailing_blanks            = 1;      // 可选
config.max_active_paths               = 4;      // 可选

KeywordSpotter kws = KeywordSpotter::Create(config);
if (!kws.Get()) {
  // 创建失败，检查路径与 config
  return -1;
}
```

## 7. 「整句匹配」能力（热词列表 ["不要说话","过来"] 时）

**需求**：说「不要过来」→ 未命中；说「过来」→ 命中。

**CXX API 已支持**：`KeywordResult` 提供 **`full_tokens`**（本段完整解码的 token 序列）。系统层只需：

1. **预加载关键词 token 表**：从 `keywords.txt` 解析出每个关键词的 token 序列（与 Python 脚本里 `load_keyword_token_sets` 一致），例如 `"过来" -> ["g","uò","l","ái"]`。
2. **每次 GetResult 后**：若 `r.keyword` 非空，则看 `r.full_tokens` 是否**完全等于**上述某条关键词的 token 序列。
   - 若 `full_tokens == 某关键词的 token 序列` → **命中**，上报该关键词。
   - 否则（例如「不要过来」时 `full_tokens` 为更长序列，不等于「过来」）→ **未命中**，不上报，仅 `Reset(&stream)` 即可。

这样即可实现：热词列表为 ["不要说话","过来"] 时，识别到「不要过来」输出未命中，识别到「过来」输出命中。

## 8. 编译与链接

与 VAD 相同：链接 sherpa-onnx 的 CXX 库（依赖 C API 与 ONNX Runtime 等）。  
可参考仓库中 `cxx-api-examples/CMakeLists.txt` 里对 `kws-cxx-api` 的配置，与 VAD 的构建方式一致。

## 9. 小结

- **接口风格**：与 [vad-cxx-api.cc](https://github.com/k2-fsa/sherpa-onnx/blob/master/cxx-api-examples/vad-cxx-api.cc) 一致，均为 CXX API，Create → CreateStream → 循环送数据 + IsReady/Decode/GetResult，命中后 Reset。
- **推荐给系统层的入口**：`cxx-api-examples/kws-cxx-api.cc`（文件输入示例） + 上述流程；流式麦克风逻辑可参考 `sherpa-onnx-keyword-spotter-microphone.cc` 的主循环。
- **若需整句匹配**：目前建议在系统层对 `GetResult().keyword` 做白名单过滤；若后续 C API 增加 `full_tokens`，再在封装层做“整句 == 某关键词”的判断即可。
