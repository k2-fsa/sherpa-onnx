# sherpa-onnx Gradle Kotlin DSL (KTS) Example

This directory demonstrates how to use [sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx) via Gradle with Kotlin DSL (`build.gradle.kts`).

## Prerequisites

- JDK 8 or above
- Gradle 8.x+ (or use the included Gradle wrapper)

## How It Works

The `build.gradle.kts` **automatically detects** your OS and architecture at build time:

```kotlin
val osName = System.getProperty("os.name").lowercase()
val osArch = System.getProperty("os.arch").lowercase()

val targetNativeClassifier = when {
    osName.contains("mac") || osName.contains("darwin") -> {
        if (osArch == "aarch64" || osArch == "arm64") "osx-aarch64" else "osx-x64"
    }
    osName.contains("linux") -> {
        if (osArch == "aarch64" || osArch == "arm64") "linux-aarch64" else "linux-x64"
    }
    osName.contains("win") -> "win-x64"
    else -> throw GradleException("Unsupported OS: $osName, Arch: $osArch")
}
```

This means:
- **No manual configuration needed** — just run `./gradlew build`
- **Works on any platform** — macOS, Linux, Windows (x64 and ARM64)
- **No CI scripts to modify** — the build file handles everything

## Dependencies

The project uses [JitPack](https://jitpack.io/) to fetch sherpa-onnx artifacts:

```kotlin
dependencies {
    // 1. JVM core API
    implementation("com.github.k2-fsa.sherpa-onnx:sherpa-onnx-jvm:refactor-jar-SNAPSHOT")

    // 2. Platform native lib (auto-detected)
    implementation("com.github.k2-fsa.sherpa-onnx:sherpa-onnx-native-lib-$targetNativeClassifier:refactor-jar-SNAPSHOT")
}
```

## Build

```bash
cd java-api-examples/gradle-kts-examples

# Using Gradle wrapper (recommended)
./gradlew build

# Or using system Gradle
gradle build
```

The build output will show the auto-detected platform:

```
--> Auto-detected platform native lib: osx-aarch64
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

## Available Native Lib Artifacts

| Platform | Artifact |
|---|---|
| macOS ARM64 | `sherpa-onnx-native-lib-osx-aarch64` |
| macOS x64 | `sherpa-onnx-native-lib-osx-x64` |
| Linux x64 | `sherpa-onnx-native-lib-linux-x64` |
| Linux ARM64 | `sherpa-onnx-native-lib-linux-aarch64` |
| Windows x64 | `sherpa-onnx-native-lib-win-x64` |

## Appendix: Groovy vs Kotlin DSL comparison

| Feature | Groovy (`build.gradle`) | Kotlin DSL (`build.gradle.kts`) |
|---|---|---|
| Syntax | Dynamic, concise | Static, type-safe |
| IDE support | Good | Excellent (auto-completion, refactoring) |
| Strings | `'single'` or `"double"` | `"double"` only |
| Function calls | `implementation '...'` | `implementation("...")` |
| Properties | `mainClass = '...'` | `mainClass.set("...")` |
| Recommended for | Legacy projects | New projects |
