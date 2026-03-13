# User Guide

*Applicable to Windows / macOS / Linux (using Windows as an example for dynamic library loading)*

## 1. Prerequisites

* Java 1.8+ environment
* Download and prepare the following:

  * Sherpa-ONNX Java API (Maven dependency)
  * Kokoro TTS model files (including `model.onnx`, etc.)

---

## 2. Add Maven Dependency

In your `pom.xml`, add:

```xml
<dependency>
  <groupId>com.litongjava</groupId>
  <artifactId>sherpa-onnx-java-api</artifactId>
  <version>1.0.1</version>
</dependency>
```

---

## 3. Obtain and Configure Native Dynamic Libraries (JNI)

### 3.1 Install ONNX Runtime

#### Windows 10

Starting from Windows 10 v1809 and all versions of Windows 11, the system comes with built-in ONNX Runtime as part of Windows ML (WinRT API), exposed through Windows.AI.MachineLearning.dll. You can directly use WinML to load and run ONNX models without additional downloads or installations.
[run-onnx-models](https://learn.microsoft.com/en-us/windows/ai/new-windows-ml/run-onnx-models)

#### Linux

Sherpa-ONNX does **not** bundle ONNX Runtime. To install it manually:

1. Download the Linux x64 binary from Microsoft’s GitHub Releases:

   ```bash
   wget https://github.com/microsoft/onnxruntime/releases/download/v1.23.2/onnxruntime-linux-x64-1.23.2.tgz
   tar -xzf onnxruntime-linux-x64-1.23.2.tgz
   ```

2. Copy and symlink the library into a system directory:

   ```bash
   sudo cp onnxruntime-linux-x64-1.23.2/lib/libonnxruntime.so* /usr/local/lib/
   sudo ln -sf /usr/local/lib/libonnxruntime.so.1.23.2 /usr/local/lib/libonnxruntime.so
   ```

3. Update the shared-library cache and verify:

   ```bash
   sudo ldconfig
   ldconfig -p | grep onnxruntime
   ```

#### macOS

Sherpa-ONNX also requires you to install ONNX Runtime on macOS:

1. Download the macOS ARM64 binary:

   ```bash
   wget https://github.com/microsoft/onnxruntime/releases/download/v1.23.2/onnxruntime-osx-arm64-1.23.2.tgz
   tar -xzf onnxruntime-osx-arm64-1.23.2.tgz
   ```

2. Copy the dylib into `/usr/local/lib`:

   ```bash
   sudo cp onnxruntime-osx-arm64-1.23.2/lib/libonnxruntime.1.23.2.dylib /usr/local/lib/
   ```

3. Add `/usr/local/lib` to `dyld`’s search path:

   ```bash
   export DYLD_LIBRARY_PATH=/usr/local/lib:$DYLD_LIBRARY_PATH
   ```

4. Verify with `otool`:

   ```bash
   otool -L /Users/ping/lib/darwin_arm64/libsherpa-onnx-jni.dylib
   ```

---

### 3.2 Common Errors & Troubleshooting

**Error Example:**

```text
Exception in thread "main" java.lang.UnsatisfiedLinkError: no sherpa-onnx-jni in java.library.path: ...
```

This means the JVM couldn’t locate the native library in `java.library.path`.

**Troubleshooting steps:**

1. Ensure you downloaded the build matching your OS and architecture (e.g. win-x64 vs. arm64).

2. Test with an absolute path:

   ```bash
   java -Djava.library.path=C:\full\path\to\jni -jar your-app.jar
   ```

3. Print or inspect `java.library.path` at runtime (e.g. `System.out.println(System.getProperty("java.library.path"));`).

4. **Do not** hack the internal `sys_paths` via reflection (it may throw `NoSuchFieldException`). Use `-Djava.library.path` instead.

---

## 4. Download & Prepare the Kokoro Model

Fetch the model package from the official release (example: Kokoro v0.19 English):

```
https://k2-fsa.github.io/sherpa/onnx/tts/pretrained_models/kokoro.html
```

```bash
# Download (manually or via script)
wget https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/kokoro-en-v0_19.tar.bz2

# Extract
tar -xjf kokoro-en-v0_19.tar.bz2

# Inspect
ls -lh kokoro-en-v0_19/
```

You should see:

```
LICENSE
README.md
espeak-ng-data/    # speech data directory
model.onnx         # TTS model
tokens.txt         # token mapping
voices.bin         # voice embeddings
```

Make sure your Java code points to these files (using either relative or absolute paths).

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
    String model   = "./kokoro-en-v0_19/model.onnx";
    String voices  = "./kokoro-en-v0_19/voices.bin";
    String tokens  = "./kokoro-en-v0_19/tokens.txt";
    String dataDir = "./kokoro-en-v0_19/espeak-ng-data";
    String text    = "Today as always, men fall into two groups: slaves and free men. Whoever does not have"
                   + " two-thirds of his day for himself, is a slave, whatever he may be: a statesman, a"
                   + " businessman, an official, or a scholar.";

    OfflineTtsKokoroModelConfig kokoroConfig = OfflineTtsKokoroModelConfig.builder()
        .setModel(model)
        .setVoices(voices)
        .setTokens(tokens)
        .setDataDir(dataDir)
        .build();

    OfflineTtsModelConfig modelConfig = OfflineTtsModelConfig.builder()
        .setKokoro(kokoroConfig)
        .setNumThreads(2)
        .setDebug(true)
        .build();

    OfflineTtsConfig config = OfflineTtsConfig.builder()
        .setModel(modelConfig)
        .build();

    OfflineTts tts = new OfflineTts(config);

    int sid   = 0;
    float speed = 1.0f;
    long start = System.currentTimeMillis();
    GeneratedAudio audio = tts.generate(text, sid, speed);
    long stop  = System.currentTimeMillis();

    float elapsed   = (stop - start) / 1000.0f;
    float duration  = audio.getSamples().length / (float) audio.getSampleRate();
    float rtf       = elapsed / duration;

    String outFile = "tts-kokoro-en.wav";
    audio.save(outFile);

    System.out.printf("-- elapsed           : %.3f seconds%n", elapsed);
    System.out.printf("-- audio duration    : %.3f seconds%n", duration);
    System.out.printf("-- real-time factor  : %.3f%n", rtf);
    System.out.printf("-- text              : %s%n", text);
    System.out.printf("-- Saved to          : %s%n", outFile);

    tts.release();
  }
}
```

### Output Explanation

After successful execution, you should see something like:

```
-- elapsed           : 6.739 seconds
-- audio duration    : 6.739 seconds
-- real-time factor  : 0.563
-- text              : ...
-- Saved to          : tts-kokoro-en.wav
```

A file named `tts-kokoro-en.wav` will appear in the current directory—play it with any audio player to verify.
