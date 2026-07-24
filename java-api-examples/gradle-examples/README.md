# sherpa-onnx Gradle Example

This directory demonstrates how to use [sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx) via Gradle.

## Prerequisites

- JDK 8 or above
- Gradle 8.x (or use the included Gradle wrapper)

## Dependencies

The project uses [JitPack](https://jitpack.io/) to fetch sherpa-onnx artifacts.
There are two ways to declare the dependencies.

### Approach 1: Simple (single dependency)

This is the easiest way — one dependency pulls in everything:

```groovy
repositories {
    mavenCentral()
    maven { url 'https://jitpack.io' }
}

dependencies {
    implementation 'com.github.k2-fsa:sherpa-onnx:refactor-jar-SNAPSHOT'
}
```

### Approach 2: Multi-module (recommended)

This is the recommended way for production use. It splits the JVM API and native libs
so you only ship the platform you need, keeping the final jar lightweight:

```groovy
repositories {
    mavenCentral()
    maven { url 'https://jitpack.io' }
}

dependencies {
    // 1. JVM core API
    implementation 'com.github.k2-fsa.sherpa-onnx:sherpa-onnx-jvm:refactor-jar-SNAPSHOT'

    // 2. Platform native lib — pick ONE for your target platform
    implementation 'com.github.k2-fsa.sherpa-onnx:sherpa-onnx-native-lib-osx-aarch64:refactor-jar-SNAPSHOT'
}
```

Available native lib artifacts for each platform:

| Platform | Artifact |
|---|---|
| macOS ARM64 | `sherpa-onnx-native-lib-osx-aarch64` |
| macOS x64 | `sherpa-onnx-native-lib-osx-x64` |
| Linux x64 | `sherpa-onnx-native-lib-linux-x64` |
| Linux ARM64 | `sherpa-onnx-native-lib-linux-aarch64` |
| Windows x64 | `sherpa-onnx-native-lib-win-x64` |
| Windows ARM64 | `sherpa-onnx-native-lib-win-arm64` |

> **Note:** The `build.gradle` in this directory uses Approach 2 by default. Approach 1 is
> included as a comment for reference.

## Build

```bash
cd java-api-examples/gradle-examples

# Using Gradle wrapper (recommended)
./gradlew build

# Or using system Gradle
gradle build
```

## Run

```bash
# Using Gradle wrapper (recommended)
./gradlew run

# Or using system Gradle
gradle run
```

Expected output:

```
sherpa-onnx version: x.y.z
sherpa-onnx gitSha1: ...
sherpa-onnx gitDate: ...
```

## Appendix: Approach 1 vs Approach 2 comparison (advanced)

The two approaches produce **very different** fat jars:

| | Approach 1 (simple) | Approach 2 (multi-module) |
|---|---|---|
| Jar size (compressed) | **~95 MB** | **~9 MB** |
| Uncompressed size | **~286 MB** | **~33 MB** |
| Total files | ~295 | ~253 |
| JVM class files | ~236 | ~236 |
| Native libs included | ALL platforms | macOS ARM64 only |

### Why Approach 2 is recommended

Approach 2 is **~10x smaller** (~9 MB vs ~95 MB) because it only ships the native libs for
your target platform. This matters for:

- **Download size**: users only download what they need
- **Startup time**: the JVM loads fewer native libraries
- **Disk space**: especially important for CI/CD and container images

### Verify it yourself

```bash
# Build and check the jar
./gradlew build
ls -lh build/libs/*.jar

# List native libs in the jar
unzip -l build/libs/sherpa-onnx-gradle-example.jar | grep -E "\.so$|\.dylib$|\.dll$"
```
