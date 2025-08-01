# User Guide

*Applicable to Windows / macOS / Linux (Windows used as example for dynamic library loading)*

## 1. Prerequisites

* Java 1.8+ runtime
* Download and prepare the following:

  * Sherpa-ONNX Java API (Maven dependency)
  * The matching platform JNI native library (e.g., `.dll` on Windows)
  * Kokoro TTS model files (including `model.onnx`, etc.)

---

## 2. Add Maven Dependency

In your `pom.xml`, add the following dependency:

```xml
<dependency>
  <groupId>com.litongjava</groupId>
  <artifactId>sherpa-onnx-java-api</artifactId>
  <version>1.0.0</version>
</dependency>
```

---

## 3. Obtain and Configure the Local Native Library (JNI)

### 3.1 Download the JNI native library for your platform

Taking Windows as an example, download the prebuilt JNI library from Hugging Face:
`https://huggingface.co/csukuangfj/sherpa-onnx-libs/tree/main/jni`

For example:
`sherpa-onnx-v1.12.7-win-x64-jni.tar.bz2`

After extraction you’ll get `sherpa-onnx-jni.dll` (on Linux it’s `.so`, on macOS it’s `.dylib`).

### 3.2 Place the native library and make it discoverable by the JVM

The JVM locates native libraries via `java.library.path`. Common approaches:

#### Option 1: Put the `.dll` in the current working directory at runtime

* In a development environment (e.g., IDE launch), the current working directory is typically the project root—place `sherpa-onnx-jni.dll` there.
* In a packaged JAR scenario, put the `.dll` alongside the JAR.

#### Option 2: Explicitly specify `java.library.path`

Example runtime flag (Windows):

```sh
java -Djava.library.path=. -jar your-app.jar
```

Or, if running from an IDE, set in VM options:

```
-Djava.library.path=path\to\directory\containing\sherpa-onnx-jni.dll
```

#### Option 3: Set a system environment variable (less recommended due to cross-platform inconsistency)

### 3.3 Common errors and troubleshooting

**Example error:**

```text
Exception in thread "main" java.lang.UnsatisfiedLinkError: no sherpa-onnx-jni in java.library.path: ...
```

This means the JVM could not find the native library in `java.library.path`.

Troubleshooting steps:

1. Verify you downloaded the version matching your OS and architecture (e.g., win-x64 vs arm64).

2. Test with an absolute path:

   ```sh
   java -Djava.library.path=C:\full\path\to\jni -jar your-app.jar
   ```

3. Print or inspect the contents of `java.library.path` in code (e.g., `System.getProperty("java.library.path")`).

4. **Do not** attempt to hack `java.library.path` by reflecting and modifying internal fields like `sys_paths`; instead, use the `-Djava.library.path` mechanism directly to avoid `NoSuchFieldException: sys_paths`.

---

## 4. Download and Prepare the Kokoro Model

Obtain the model package from the official release (example: English Kokoro v0.19):

```
https://k2-fsa.github.io/sherpa/onnx/tts/pretrained_models/kokoro.html
```

```sh
# Download (manually or via script)
# e.g., from GitHub releases:
# kokoro-en-v0_19.tar.bz2

# Extract
tar -xjf kokoro-en-v0_19.tar.bz2

# Inspect structure
ls -lh kokoro-en-v0_19/
```

Example directory contents after extraction (these should be present):

```
LICENSE
README.md
espeak-ng-data/        # speech data directory
model.onnx            # TTS model
tokens.txt           # token mapping
voices.bin           # voice embeddings
```

Ensure your Java program uses correct paths to these files or directories (absolute or relative).

---

## 5. Test Code (Java Example)

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

### Output Explanation

On successful execution you should see something like:

```
-- elapsed : 6.739 seconds
-- audio duration: 6.739 seconds
-- real-time factor (RTF): 0.563
-- text: ...
-- Saved to tts-kokoro-en.wav
```

And a file `tts-kokoro-en.wav` will be created in the current directory; you can play it with any audio player to verify.
