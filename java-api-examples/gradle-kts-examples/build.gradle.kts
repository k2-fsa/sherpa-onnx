plugins {
    application
    java
}

application {
    mainClass.set("com.k2fsa.sherpa.onnx.example.VersionTest")
}

repositories {
    mavenCentral()
    maven { url = uri("https://jitpack.io") }
}

dependencies {
    // ============================================================
    // Approach 1: Simple — one dependency pulls in everything.
    // ============================================================
    // implementation("com.github.k2-fsa:sherpa-onnx:refactor-jar-SNAPSHOT")

    // ============================================================
    // Approach 2 (recommended): Multi-module — split JVM API and
    // native libs so you only ship the platform you need.
    // ============================================================

    // 1. JVM core API Jar
    implementation("com.github.k2-fsa.sherpa-onnx:sherpa-onnx-jvm:refactor-jar-SNAPSHOT")

    // 2. Platform native lib — uncomment ONE for your target platform

    // macOS ARM64 (Apple Silicon)
    implementation("com.github.k2-fsa.sherpa-onnx:sherpa-onnx-native-lib-osx-aarch64:refactor-jar-SNAPSHOT")

    // macOS x64 (Intel)
    // implementation("com.github.k2-fsa.sherpa-onnx:sherpa-onnx-native-lib-osx-x64:refactor-jar-SNAPSHOT")

    // Linux x64
    // implementation("com.github.k2-fsa.sherpa-onnx:sherpa-onnx-native-lib-linux-x64:refactor-jar-SNAPSHOT")

    // Linux ARM64
    // implementation("com.github.k2-fsa.sherpa-onnx:sherpa-onnx-native-lib-linux-aarch64:refactor-jar-SNAPSHOT")

    // Windows x64
    // implementation("com.github.k2-fsa.sherpa-onnx:sherpa-onnx-native-lib-win-x64:refactor-jar-SNAPSHOT")
}

java {
    sourceCompatibility = JavaVersion.VERSION_1_8
    targetCompatibility = JavaVersion.VERSION_1_8
}

tasks.jar {
    manifest {
        attributes("Main-Class" to "com.k2fsa.sherpa.onnx.example.VersionTest")
    }
    duplicatesStrategy = DuplicatesStrategy.EXCLUDE
    from(configurations.runtimeClasspath.get().map { if (it.isDirectory) it else zipTree(it) })
}
