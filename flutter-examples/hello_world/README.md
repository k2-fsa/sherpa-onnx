# hello_world

A minimal Flutter example that displays sherpa-onnx version information.

## How this demo was created

### Step 1: Create the Flutter project

```bash
cd flutter-examples
flutter create --project-name hello_world --org com.k2fsa hello_world
```

### Step 2: Add sherpa_onnx dependency

Edit `pubspec.yaml`, replace the `dependencies:` section with:

```yaml
dependencies:
  flutter:
    sdk: flutter
  sherpa_onnx: ^1.13.4
```

Then run:

```bash
cd hello_world
flutter pub get
```

### Step 3: Replace lib/main.dart

Replace the contents of `lib/main.dart` with:

```dart
import 'package:flutter/material.dart';
import 'package:sherpa_onnx/sherpa_onnx.dart';

void main() {
  WidgetsFlutterBinding.ensureInitialized();
  initBindings();
  runApp(const MyApp());
}

class MyApp extends StatelessWidget {
  const MyApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'sherpa-onnx hello world',
      home: const VersionPage(),
    );
  }
}

class VersionPage extends StatelessWidget {
  const VersionPage({super.key});

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text('sherpa-onnx hello world')),
      body: Center(
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            Text('sherpa-onnx version: ${getVersion()}'),
            Text('Git SHA1: ${getGitSha1()}'),
            Text('Git date: ${getGitDate()}'),
            Text('onnxruntime version: ${getOnnxruntimeVersion()}'),
          ],
        ),
      ),
    );
  }
}
```

### Step 4: Run the app

```bash
flutter run
```

The app will display the sherpa-onnx version, git SHA1, git date, and
onnxruntime version.

## Platform notes

- **macOS**: Minimum deployment target is 10.15
- **iOS**: Minimum deployment target is 13.0
- **Android**: Minimum SDK 21 (default)
- **Linux**: x64 and aarch64 supported
- **Windows**: x64 supported

## Building for specific platforms

```bash
# macOS
flutter build macos

# iOS (no codesign for CI)
flutter build ios --no-codesign

# Android
flutter build apk

# Linux
flutter build linux

# Windows
flutter build windows
```

## Running on Web (Chrome)

```bash
cd flutter-examples/hello_world
flutter pub get
flutter run -d chrome
```

### Build for deployment

```bash
flutter build web
```

The output is in `build/web/`. Serve it with any HTTP server:

```bash
cd build/web
python3 -m http.server 8080
```

Then start your browser and visit <http://localhost:8080> .

### Notes

- The web build uses WebAssembly (via Emscripten) to run the sherpa-onnx C API
  in the browser. No server-side processing is needed.
- The WASM binary is ~15-20MB as it includes all features (ASR, TTS, VAD, etc.)
- First load may take a few seconds while the WASM module is downloaded and
  compiled

## Running on iOS simulator

### Step 1: List available simulators

```bash
xcrun simctl list devices
```

Look for a booted simulator, e.g.:

```
iPhone 16 Plus (UUID) (Booted)
```

### Step 2: Run on the simulator

```bash
flutter run -d <UUID>
```

For example:

```bash
flutter run -d 34FB0674-4ABA-4870-ABF2-D0D6E110A7C2
```

### Troubleshooting

If you see `ld: framework 'sherpa_onnx' not found`, it means the Xcode
project has stale Swift Package Manager (SPM) references. Remove them:

```bash
cd ios
# Remove SPM references from the Xcode project
# (edit Runner.xcodeproj/project.pbxproj to remove FlutterGeneratedPluginSwiftPackage entries)
```

Or regenerate the iOS project:

```bash
flutter clean
flutter pub get
flutter run -d <UUID>
```
