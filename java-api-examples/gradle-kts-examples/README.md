# sherpa-onnx Gradle Kotlin DSL (KTS) Example

This directory demonstrates how to use [sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx) via Gradle with Kotlin DSL (`build.gradle.kts`).

## Prerequisites

- JDK 8 or above
- Gradle 8.x+ (or use the included Gradle wrapper)

## Dependencies

The project uses [JitPack](https://jitpack.io/) to fetch sherpa-onnx artifacts.
There are two ways to declare the dependencies.

### Approach 1: Simple (single dependency)

This is the easiest way — one dependency pulls in everything:

```kotlin
repositories {
    mavenCentral()
    maven { url = uri("https://jitpack.io") }
}

dependencies {
    implementation("com.github.k2-fsa:sherpa-onnx:refactor-jar-SNAPSHOT")
}
```

### Approach 2: Multi-module (recommended)

This is the recommended way for production use. It splits the JVM API and native libs
so you only ship the platform you need, keeping the final jar lightweight:

```kotlin
repositories {
    mavenCentral()
    maven { url = uri("https://jitpack.io") }
}

dependencies {
    // 1. JVM core API
    implementation("com.github.k2-fsa.sherpa-onnx:sherpa-onnx-jvm:refactor-jar-SNAPSHOT")

    // 2. Platform native lib — pick ONE for your target platform
    implementation("com.github.k2-fsa.sherpa-onnx:sherpa-onnx-native-lib-osx-aarch64:refactor-jar-SNAPSHOT")
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

> **Note:** The `build.gradle.kts` in this directory uses Approach 2 by default. Approach 1 is
> included as a comment for reference.

## Build

```bash
cd java-api-examples/gradle-kts-examples

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

## Appendix: Groovy vs Kotlin DSL comparison

| Feature | Groovy (`build.gradle`) | Kotlin DSL (`build.gradle.kts`) |
|---|---|---|
| Syntax | Dynamic, concise | Static, type-safe |
| IDE support | Good | Excellent (auto-completion, refactoring) |
| Strings | `'single'` or `"double"` | `"double"` only |
| Function calls | `implementation '...'` | `implementation("...")` |
| Properties | `mainClass = '...'` | `mainClass.set("...")` |
| Recommended for | Legacy projects | New projects |

## Appendix: Approach 1 vs Approach 2 comparison (advanced)

The two approaches produce **very different** fat jars:

| | Approach 1 (simple) | Approach 2 (multi-module) |
|---|---|---|
| Jar size (compressed) | **~95 MB** | **~9 MB** |
| Uncompressed size | **~286 MB** | **~33 MB** |
| Native libs included | ALL platforms | Target platform only |

### Why Approach 2 is recommended

Approach 2 is **~10x smaller** because it only ships the native libs for your target platform.
