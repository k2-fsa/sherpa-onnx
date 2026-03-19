# KWS SDK — 对外封装库

将 KWS（关键词检测）封装为独立 API，便于提供给第三方集成，**无需交付整个 sherpa-onnx 源码**。

## 提供给对方的文件

1. **头文件**：`include/kws_api.h`（唯一需要包含的头文件，不暴露 sherpa-onnx 类型）
2. **库文件**：构建得到的 `libkws_sdk.a`（或 `kws_sdk.lib` / `libkws_sdk.so`）
3. **依赖**：对方还需链接 **sherpa-onnx** 的库（如 `sherpa-onnx-cxx-api`、`sherpa-onnx-c-api`、`sherpa-onnx-core` 及 ONNX Runtime 等）。可由你在本仓库一次编出后，将以下产物一并打包给对方：
   - `libkws_sdk.a`
   - `libsherpa-onnx-cxx-api.a`、`libsherpa-onnx-c-api.a`、`libsherpa-onnx-core.a`
   - 以及 ONNX Runtime 等依赖（或说明由对方自行安装/链接）

## 在本仓库中构建

在项目根目录：

```bash
mkdir build && cd build
cmake .. -DBUILD_SHARED_LIBS=OFF   # 可选，静态库便于分发
cmake --build .
```

产物：

- 静态库：`build/lib/libkws_sdk.a`（或 `build/kws-sdk/libkws_sdk.a`，视 CMake 配置而定）

## 对方项目中的用法

1. **包含头文件**：`#include "kws_api.h"`
2. **链接**：链接 `kws_sdk` 以及 sherpa-onnx 相关库（见上）。
3. **调用流程**（与 VAD 类似）：
   - `Kws::Create(config)` 初始化一次；
   - `Kws::CreateStream(engine)` 创建流；
   - 循环：送 PCM → `AcceptWaveform` → `IsReady` / `Decode` / `GetResult`，命中后 `Reset`；
   - 用 `Result::full_tokens` 与本地关键词 token 表做整句匹配，决定是否算真正命中。

详见 `kws_api.h` 内注释及 `cxx-api-examples/KWS_CXX_API_接口说明.md`。
