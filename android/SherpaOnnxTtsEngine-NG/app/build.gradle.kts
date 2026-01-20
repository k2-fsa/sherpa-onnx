import java.text.SimpleDateFormat
import java.util.Date
import java.util.TimeZone

plugins {
    id("com.android.application")
    id("org.jetbrains.kotlin.android")
    id("kotlinx-serialization")

}

val buildTime: Long
    get() {
        val t = Date().time / 1000
        return t
    }

val releaseTime: String
    get() {
        val sdf = SimpleDateFormat("yy.MMddHH")
        sdf.timeZone = TimeZone.getTimeZone("GMT+8")
        return sdf.format(Date())
    }

val version = "1.$releaseTime"

val gitCommits: Int
    get() {
        val process = ProcessBuilder("git", "rev-list", "HEAD", "--count").start()
        return process.inputStream.reader().use { it.readText() }.trim().toInt()
    }
android {
    namespace = "com.k2fsa.sherpa.onnx.tts.engine"
    compileSdk = 34

    defaultConfig {
        applicationId = "com.k2fsa.sherpa.onnx.tts.engine"
        minSdk = 21
        targetSdk = 34
        versionCode = gitCommits
        versionName = version

        testInstrumentationRunner = "androidx.test.runner.AndroidJUnitRunner"
        vectorDrawables {
            useSupportLibrary = true
        }
    }

    signingConfigs {
        create("release") {
            storeFile = file("sherpa-onnx.jks")
            storePassword = "sherpa-onnx"
            keyAlias = "sherpa-onnx"
            keyPassword = "sherpa-onnx"
        }
    }

    buildTypes {
        release {
            signingConfig = signingConfigs["release"]
            isMinifyEnabled = false
            proguardFiles(
                getDefaultProguardFile("proguard-android-optimize.txt"),
                "proguard-rules.pro"
            )
        }

        debug {
            applicationIdSuffix = ".debug"
            versionNameSuffix = "_debug"
        }
    }
    compileOptions {
        isCoreLibraryDesugaringEnabled = true

        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }
    kotlinOptions {
        jvmTarget = "17"
    }
    buildFeatures {
        compose = true
    }
    composeOptions {
        kotlinCompilerExtensionVersion = "1.5.8"
    }
    packaging {
        resources {
            excludes += "/META-INF/{AL2.0,LGPL2.1}"
        }
    }
}

dependencies {
    coreLibraryDesugaring("com.android.tools:desugar_jdk_libs:2.0.4")

    implementation("androidx.localbroadcastmanager:localbroadcastmanager:1.1.0")

    implementation("com.squareup.okio:okio:3.3.0")
    implementation("com.squareup.okhttp3:okhttp:4.11.0")
    implementation("com.github.liangjingkanji:Net:3.6.4")

    implementation("androidx.navigation:navigation-compose:2.7.7")
    implementation("androidx.documentfile:documentfile:1.0.1")

    implementation("org.apache.commons:commons-compress:1.26.0")
    implementation("commons-codec:commons-codec:1.16.1")

    implementation("org.tukaani:xz:1.9")


    implementation("com.geyifeng.immersionbar:immersionbar:3.2.2")
    implementation("com.geyifeng.immersionbar:immersionbar-ktx:3.2.2")

    implementation("com.charleskorn.kaml:kaml:0.57.0")
    implementation("org.jetbrains.kotlinx:kotlinx-serialization-json:1.6.2")

    implementation("com.github.FunnySaltyFish.ComposeDataSaver:data-saver:v1.1.5")
    implementation("org.burnoutcrew.composereorderable:reorderable:0.9.6")

    val accompanistVersion = "0.34.0"
    implementation("com.google.accompanist:accompanist-systemuicontroller:${accompanistVersion}")
//    implementation("com.google.accompanist:accompanist-navigation-animation:${accompanistVersion}")
    implementation("com.google.accompanist:accompanist-permissions:${accompanistVersion}")


    implementation("androidx.constraintlayout:constraintlayout-compose:1.0.1")
    implementation("androidx.navigation:navigation-compose:2.7.7")

    implementation("io.github.dokar3:sheets-m3:0.5.4")

    val composeBom = platform("androidx.compose:compose-bom:2024.02.01")
    implementation("androidx.core:core-ktx:1.12.0")
    implementation("androidx.lifecycle:lifecycle-runtime-ktx:2.7.0")
    implementation("androidx.activity:activity-compose:1.8.2")
    implementation(composeBom)
    implementation("androidx.compose.ui:ui")
    implementation("androidx.compose.ui:ui-graphics")
    implementation("androidx.compose.ui:ui-tooling-preview")
    implementation("androidx.compose.material3:material3")
    implementation("androidx.appcompat:appcompat:1.6.1")

//    implementation("androidx.compose.material3:material-icons-core")
    implementation("androidx.compose.material:material-icons-extended")
    implementation("androidx.lifecycle:lifecycle-viewmodel-compose:2.7.0")

    testImplementation("junit:junit:4.13.2")
    androidTestImplementation("androidx.test.ext:junit:1.1.5")
    androidTestImplementation("androidx.test.espresso:espresso-core:3.5.1")
    androidTestImplementation(composeBom)
    androidTestImplementation("androidx.compose.ui:ui-test-junit4")
    debugImplementation("androidx.compose.ui:ui-tooling")
    debugImplementation("androidx.compose.ui:ui-test-manifest")
}