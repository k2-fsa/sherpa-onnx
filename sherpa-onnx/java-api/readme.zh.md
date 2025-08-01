# 使用指南

*适用于 Windows / macOS / Linux（以 Windows 为例说明动态库加载）*

## 1. 前提条件

* Java 1.8+ 环境
* 下载并准备好以下内容：
  * Sherpa-ONNX Java API（Maven 依赖）
  * 对应平台的 JNI 动态链接库（如 Windows 为 `.dll`）
  * Kokoro TTS 模型文件（包含 `model.onnx` 等）

---

## 2. 添加 Maven 依赖

在你的 `pom.xml` 中添加如下依赖：

```xml
<dependency>
  <groupId>com.litongjava</groupId>
  <artifactId>sherpa-onnx-java-api</artifactId>
  <version>1.0.0</version>
</dependency>
```

---

## 3. 获取并配置本地动态链接库（JNI）

### 3.1 下载对应平台的 JNI 动态库

以 Windows 为例，从 Hugging Face 下载预编译的 JNI 库：
`https://huggingface.co/csukuangfj/sherpa-onnx-libs/tree/main/jni`

例如：
`sherpa-onnx-v1.12.7-win-x64-jni.tar.bz2`

解压后会得到 `sherpa-onnx-jni.dll`（Linux 是 `.so`，macOS 是 `.dylib`）。

### 3.2 放置动态库并让 JVM 找到它

JVM 通过 `java.library.path` 查找本地 JNI 库，有几种常用做法：

#### 方案一：把 `.dll` 放在运行时的当前目录

* 如果你在开发环境运行（比如 IDE 启动），当前工作目录通常是工程根目录，把 `sherpa-onnx-jni.dll` 放那儿即可。
* 如果运行的是打包后的 JAR（生产环境），把 `.dll` 和 JAR 放在同一目录。

#### 方案二：显式指定 `java.library.path`

运行时加参数示例（Windows 示例）：

```sh
java -Djava.library.path=. -jar your-app.jar
```

或者如果从 IDE 运行，在 VM options 里填入：

```
-Djava.library.path=路径到包含sherpa-onnx-jni.dll的目录
```

#### 方案三：设置系统环境变量（不推荐因为跨平台不一致）

### 3.3 常见错误与排查

**错误示例：**

```text
Exception in thread "main" java.lang.UnsatisfiedLinkError: no sherpa-onnx-jni in java.library.path: ...
```

说明 JVM 没有在 `java.library.path` 中找到本地库。

排查步骤：

1. 确认下载的是与你操作系统与架构匹配的版本（如 win-x64 vs arm64 等）。
2. 用绝对路径测试：将 `.dll` 放在某个目录并运行：

   ```sh
   java -Djava.library.path=C:\full\path\to\jni -jar your-app.jar
   ```
3. 打印或检查 `java.library.path` 内容（示例代码里可输出 `System.getProperty("java.library.path")`）。
4. 避免通过反射修改 `sys_paths`（不要尝试 hack `java.library.path` 的内部字段，容易引发 `NoSuchFieldException: sys_paths`，建议直接用 `-Djava.library.path`）。

---

## 4. 下载并准备 Kokoro 模型

从官方 release 获取模型包（以英文 Kokoro v0.19 为例）：
```
https://k2-fsa.github.io/sherpa/onnx/tts/pretrained_models/kokoro.html
```

```sh
# 下载（手工或脚本）
# 例如从 GitHub releases:
# kokoro-en-v0_19.tar.bz2

# 解压
tar -xjf kokoro-en-v0_19.tar.bz2

# 查看结构
ls -lh kokoro-en-v0_19/
```

该目录结构示例（解压后应包含）：

```
LICENSE
README.md
espeak-ng-data/        # 语音数据目录
model.onnx            # TTS 模型
tokens.txt           # token 映射
voices.bin           # voice embedding
```

确保这些路径在你的 Java 程序中指向正确的位置（相对或绝对皆可）。

---

## 5. 测试代码（Java 示例）

```java
package com.litongjava.linux.tts;

import com.k2fsa.sherpa.onnx.GeneratedAudio;
import com.k2fsa.sherpa.onnx.OfflineTts;
import com.k2fsa.sherpa.onnx.OfflineTtsConfig;
import com.k2fsa.sherpa.onnx.OfflineTtsKokoroModelConfig;
import com.k2fsa.sherpa.onnx.OfflineTtsModelConfig;

public class NonStreamingTtsKokoroEn {
  public static void main(String[] args) {
    String model = "./kokoro-en-v0_19/model.onnx";
    String voices = "./kokoro-en-v0_19/voices.bin";
    String tokens = "./kokoro-en-v0_19/tokens.txt";
    String dataDir = "./kokoro-en-v0_19/espeak-ng-data";
    String text = "Today as always, men fall into two groups: slaves and free men. Whoever does not have"
        + " two-thirds of his day for himself, is a slave, whatever he may be: a statesman, a"
        + " businessman, an official, or a scholar.";

    OfflineTtsKokoroModelConfig kokoroModelConfig = OfflineTtsKokoroModelConfig.builder()
        .setModel(model)
        .setVoices(voices)
        .setTokens(tokens)
        .setDataDir(dataDir)
        .build();

    OfflineTtsModelConfig modelConfig = OfflineTtsModelConfig.builder()
        .setKokoro(kokoroModelConfig)
        .setNumThreads(2)
        .setDebug(true)
        .build();

    OfflineTtsConfig config = OfflineTtsConfig.builder()
        .setModel(modelConfig)
        .build();

    OfflineTts tts = new OfflineTts(config);

    int sid = 0;
    float speed = 1.0f;
    long start = System.currentTimeMillis();
    GeneratedAudio audio = tts.generate(text, sid, speed);
    long stop = System.currentTimeMillis();

    float timeElapsedSeconds = (stop - start) / 1000.0f;
    float audioDuration = audio.getSamples().length / (float) audio.getSampleRate();
    float real_time_factor = timeElapsedSeconds / audioDuration;

    String waveFilename = "tts-kokoro-en.wav";
    audio.save(waveFilename);
    System.out.printf("-- elapsed : %.3f seconds\n", timeElapsedSeconds);
    System.out.printf("-- audio duration: %.3f seconds\n", audioDuration);
    System.out.printf("-- real-time factor (RTF): %.3f\n", real_time_factor);
    System.out.printf("-- text: %s\n", text);
    System.out.printf("-- Saved to %s\n", waveFilename);

    tts.release();
  }
}
```

### 输出说明

成功执行后会输出类似：

```
-- elapsed : 6.739 seconds
-- audio duration: 6.739 seconds
-- real-time factor (RTF): 0.563
-- text: ...
-- Saved to tts-kokoro-en.wav
```

并在当前目录生成 `tts-kokoro-en.wav`，可以用任意音频播放器播放验证。

---