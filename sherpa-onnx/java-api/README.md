# sherpa-onnx JVM API

This is the JVM (Java) API for [sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx).

## Prerequisites

- JDK 8 or above
- Maven 3.x (or use the included wrapper if available)

## Project Structure

```
sherpa-onnx/java-api/
├── pom.xml                    # Maven build configuration
└── src/main/java/com/k2fsa/sherpa/onnx/
    ├── LibraryLoader.java
    ├── VersionInfo.java
    ├── OnlineRecognizer.java
    ├── OfflineRecognizer.java
    └── ... (all Java source files)
```

## Build

```bash
cd sherpa-onnx/java-api

# Clean and build
mvn clean package

# Build without clean
mvn package

# Install to local Maven repository (~/.m2)
mvn install
```

The generated jar will be in `target/sherpa-onnx-jvm-<version>.jar`.

## Clean

```bash
mvn clean
```

## Output

After building, you will find:

| File | Description |
|---|---|
| `target/sherpa-onnx-jvm-1.13.4.jar` | JVM API jar (classes only) |

## Usage

### In a Maven Project

Add to your `pom.xml`:

```xml
<dependency>
    <groupId>com.github.k2-fsa.sherpa-onnx</groupId>
    <artifactId>sherpa-onnx-jvm</artifactId>
    <version>1.13.4</version>
</dependency>
```

### In a Gradle Project

Add to your `build.gradle`:

```groovy
implementation 'com.github.k2-fsa.sherpa-onnx:sherpa-onnx-jvm:1.13.4'
```

### In a Gradle Kotlin DSL Project

Add to your `build.gradle.kts`:

```kotlin
implementation("com.github.k2-fsa.sherpa-onnx:sherpa-onnx-jvm:1.13.4")
```

### From Command Line

```bash
# Compile your code against the jar
javac -cp sherpa-onnx/java-api/target/sherpa-onnx-jvm-1.13.4.jar YourApp.java

# Run your code (with JNI library path)
java -Djava.library.path=/path/to/jni/libs -cp sherpa-onnx/java-api/target/sherpa-onnx-jvm-1.13.4.jar YourApp
```

## Native Libraries

This jar only contains the JVM API classes. You also need platform-specific native libraries
to run the application. See the [main README](../../README.md) for details on how to obtain
native libraries for your platform.

### Native Library Artifacts (from JitPack)

| Platform | Artifact |
|---|---|
| macOS ARM64 | `com.github.k2-fsa.sherpa-onnx:sherpa-onnx-native-lib-osx-aarch64` |
| macOS x64 | `com.github.k2-fsa.sherpa-onnx:sherpa-onnx-native-lib-osx-x64` |
| Linux x64 | `com.github.k2-fsa.sherpa-onnx:sherpa-onnx-native-lib-linux-x64` |
| Linux ARM64 | `com.github.k2-fsa.sherpa-onnx:sherpa-onnx-native-lib-linux-aarch64` |
| Windows x64 | `com.github.k2-fsa.sherpa-onnx:sherpa-onnx-native-lib-win-x64` |

## Adding New Java Files

With Maven, you don't need to edit any build configuration when adding new Java files.
Simply place your new `.java` file in:

```
src/main/java/com/k2fsa/sherpa/onnx/YourNewClass.java
```

Maven will automatically discover and compile it.

## Version

The current version is `1.13.4` (defined in `pom.xml`).

To update the version:

1. Edit `pom.xml` and change the `<version>` tag
2. Rebuild with `mvn clean package`
