# Introduction

This directory contains flutter examples of `sherpa-onnx`.

# Ways to create an example
```bash
flutter create --platforms windows,macos streaming_asr
cd streaming_asr
flutter pub get

# to support a new platform, e.g., android, use

cd streaming_asr
flutter create --platforms --org com.k2fsa.sherpa.onnx android ./
```

To run with android, first use
```
(py38) fangjuns-MacBook-Pro:streaming_asr fangjun$ flutter run devices
No devices found yet. Checking for wireless devices...

No supported devices found with name or id matching 'android-arm64'.

The following devices were found:
Mi 10 (mobile)  • 61106679 • android-arm64  • Android 12 (API 31)
macOS (desktop) • macos    • darwin-x64     • macOS 13.1 22C65 darwin-x64
Chrome (web)    • chrome   • web-javascript • Google Chrome 126.0.6478.127
```
to find available devices. I have attached my Android phone (Xiaomi 10) to my computer
and it shows the device ID of my Android phone is `61106679`, so I use

```bash
(py38) fangjuns-MacBook-Pro:streaming_asr fangjun$ flutter run -d 61106679
```

to run it.

If you get the following errors and hint:

```
BUILD FAILED in 2m 43s
Running Gradle task 'assembleDebug'...                            165.3s

┌─ Flutter Fix ───────────────────────────────────────────────────────────────────────────────────────────────────┐
│ The plugin record_android requires a higher Android SDK version.                                                │
│ Fix this issue by adding the following to the file                                                              │
│ /Users/fangjun/open-source/sherpa-onnx/flutter-examples/streaming_asr/android/app/build.gradle:                 │
│ android {                                                                                                       │
│   defaultConfig {                                                                                               │
│     minSdkVersion 23                                                                                            │
│   }                                                                                                             │
│ }                                                                                                               │
│                                                                                                                 │
│                                                                                                                 │
│ Following this change, your app will not be available to users running Android SDKs below 23.                   │
│ Consider searching for a version of this plugin that supports these lower versions of the Android SDK instead.  │
│ For more information, see: https://docs.flutter.dev/deployment/android#reviewing-the-gradle-build-configuration │
└─────────────────────────────────────────────────────────────────────────────────────────────────────────────────┘
Error: Gradle task assembleDebug failed with exit code 1
```

Please use the following changes:

```diff
--- a/flutter-examples/streaming_asr/android/app/build.gradle
+++ b/flutter-examples/streaming_asr/android/app/build.gradle
@@ -38,7 +38,7 @@ android {
         applicationId = "com.k2fsa.sherpa.onnx.streaming_asr"
         // You can update the following values to match your application needs.
         // For more information, see: https://docs.flutter.dev/deployment/android#reviewing-the-gradle-build-configuration.
-        minSdk = flutter.minSdkVersion
+        minSdk = 23
         targetSdk = flutter.targetSdkVersion
         versionCode = flutterVersionCode.toInteger()
         versionName = flutterVersionName
```

If you get the following errors:

```
Launching lib/main.dart on Mi 10 in debug mode...
ERROR:/Users/fangjun/open-source/sherpa-onnx/flutter-examples/streaming_asr/build/record_android/intermediates/runtime_library_classes_jar/debug/clas
ses.jar: D8: com.android.tools.r8.internal.Hc: Sealed classes are not supported as program classes

FAILURE: Build failed with an exception.

* What went wrong:
Execution failed for task ':app:mergeLibDexDebug'.
> Could not resolve all files for configuration ':app:debugRuntimeClasspath'.
   > Failed to transform classes.jar (project :record_android) to match attributes {artifactType=android-dex, asm-transformed-variant=NONE, com.andro
id.build.api.attributes.AgpVersionAttr=7.3.0, com.android.build.api.attributes.BuildTypeAttr=debug, com.android.build.gradle.internal.attributes.Vari
antAttr=debug, dexing-enable-desugaring=true, dexing-enable-jacoco-instrumentation=false, dexing-is-debuggable=true, dexing-min-sdk=23, org.gradle.ca
tegory=library, org.gradle.jvm.environment=android, org.gradle.libraryelements=jar, org.gradle.usage=java-runtime, org.jetbrains.kotlin.platform.type
=androidJvm}.
      > Execution failed for DexingWithClasspathTransform: /Users/fangjun/open-source/sherpa-onnx/flutter-examples/streaming_asr/build/record_android
/intermediates/runtime_library_classes_jar/debug/classes.jar.
         > Error while dexing.

* Try:
> Run with --stacktrace option to get the stack trace.
> Run with --info or --debug option to get more log output.
> Run with --scan to get full insights.

* Get more help at https://help.gradle.org

BUILD FAILED in 2m 10s
```

Please refer to <https://github.com/llfbandit/record/blob/master/record_android/README.md>
to make the following changes

```diff
diff --git a/flutter-examples/streaming_asr/android/settings.gradle b/flutter-examples/streaming_asr/android/settings.gradle
index 536165d3..9b1a1012 100644
--- a/flutter-examples/streaming_asr/android/settings.gradle
+++ b/flutter-examples/streaming_asr/android/settings.gradle
@@ -18,7 +18,7 @@ pluginManagement {

 plugins {
     id "dev.flutter.flutter-plugin-loader" version "1.0.0"
-    id "com.android.application" version "7.3.0" apply false
+    id "com.android.application" version "7.4.2" apply false
     id "org.jetbrains.kotlin.android" version "1.7.10" apply false
 }
```
