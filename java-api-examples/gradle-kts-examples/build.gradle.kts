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

// Auto-detect current OS and architecture
val osName = System.getProperty("os.name").lowercase()
val osArch = System.getProperty("os.arch").lowercase()

val targetNativeClassifier = when {
    osName.contains("mac") || osName.contains("darwin") -> {
        if (osArch == "aarch64" || osArch == "arm64") "osx-aarch64" else "osx-x64"
    }
    osName.contains("linux") -> {
        if (osArch == "aarch64" || osArch == "arm64") "linux-aarch64" else "linux-x64"
    }
    osName.contains("win") -> {
        if (osArch == "aarch64" || osArch == "arm64") {
            throw GradleException("Windows ARM64 is not supported yet. Please use x64.")
        }
        "win-x64"
    }
    else -> throw GradleException("Unsupported OS: $osName, Arch: $osArch")
}

logger.lifecycle("--> Auto-detected platform native lib: $targetNativeClassifier")

dependencies {
    // 1. JVM core API
    implementation("com.github.k2-fsa.sherpa-onnx:sherpa-onnx-jvm:refactor-jar-SNAPSHOT")

    // 2. Platform native lib (auto-detected)
    implementation("com.github.k2-fsa.sherpa-onnx:sherpa-onnx-native-lib-$targetNativeClassifier:refactor-jar-SNAPSHOT")
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
