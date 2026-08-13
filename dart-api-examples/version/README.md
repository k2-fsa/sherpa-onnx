# Version Example

This example shows how to initialize sherpa-onnx and print version information.
It demonstrates four different ways to initialize the library.

## Prerequisites

- [Dart SDK](https://dart.dev/get-dart) (3.0.0 or later)
- sherpa-onnx native library (installed automatically via pub dependencies)

## Quick Start

```bash
cd dart-api-examples/version
./run.sh
```

This runs all four examples below.

## Examples

### 1. Sync Init (`main_sync.dart`)

The simplest way to initialize sherpa-onnx. Calls `initBindings()` which
blocks until the native library is loaded.

```dart
import 'package:sherpa_onnx/sherpa_onnx.dart' as sherpa_onnx;

void main() {
  sherpa_onnx.initBindings();
  print('version: ${sherpa_onnx.getVersion()}');
}
```

**When to use:** Simple scripts, command-line tools, and Flutter apps where
you don't need async initialization.

**Does NOT work on web.** Use `initBindingsAsync()` for web support.

### 2. Async Init (`main_async.dart`)

Calls `initBindingsAsync()` which returns a `Future`. This is the recommended
approach because it works on all platforms including web.

```dart
import 'package:sherpa_onnx/sherpa_onnx.dart' as sherpa_onnx;

void main() async {
  await sherpa_onnx.initBindingsAsync();
  print('version: ${sherpa_onnx.getVersion()}');
}
```

**When to use:** Anywhere. This is the safest choice that works on Flutter,
Dart CLI, and web.

### 3. Isolate with Sync Init (`main_isolate_sync.dart`)

Demonstrates using sherpa-onnx in a Dart isolate with synchronous
initialization.

**Key concept:** Each isolate has its own memory space and FFI bindings.
You MUST call `initBindings()` or `initBindingsAsync()` in every isolate
that uses sherpa-onnx. Calling it in one isolate does NOT make sherpa-onnx
available in other isolates.

```dart
import 'dart:isolate';
import 'package:sherpa_onnx/sherpa_onnx.dart' as sherpa_onnx;

void worker(SendPort sendPort) {
  // MUST initialize in this isolate too!
  sherpa_onnx.initBindings();
  sendPort.send(sherpa_onnx.getVersion());
}

void main() async {
  // Initialize in the main isolate.
  sherpa_onnx.initBindings();
  print('main: ${sherpa_onnx.getVersion()}');

  // Spawn a worker isolate.
  final receivePort = ReceivePort();
  await Isolate.spawn(worker, receivePort.sendPort);
  final version = await receivePort.first;
  print('worker: $version');
}
```

**When to use:** Background processing (e.g., TTS generation, model loading)
where you want synchronous initialization in each isolate.

### 4. Isolate with Async Init (`main_isolate_async.dart`)

Same as above but uses `initBindingsAsync()` in both isolates.

```dart
void worker(SendPort sendPort) async {
  await sherpa_onnx.initBindingsAsync();  // async init
  sendPort.send(sherpa_onnx.getVersion());
}

void main() async {
  await sherpa_onnx.initBindingsAsync();  // async init
  // ... spawn isolate as before
}
```

**When to use:** Same as the sync isolate example, but when you prefer
async initialization (e.g., in Flutter apps or web-compatible code).

## How Initialization Works

When you call `initBindings()` or `initBindingsAsync()`:

1. The library detects the current platform (macOS, Linux, Windows, iOS,
   Android, or web).
2. It locates the native `sherpa-onnx-c-api` library:
   - **Flutter apps:** The library is linked into the app bundle by the
     build system. `DynamicLibrary.process()` loads it directly.
   - **Dart CLI:** The library is located in the pub cache via
     `Isolate.resolvePackageUri()`. `DynamicLibrary.open()` loads it
     from disk.
   - **Web:** The WASM module is loaded from bundled assets.
3. Once loaded, all sherpa-onnx APIs become available in that isolate.

## API Reference

| Function | Sync/Async | Flutter | Dart CLI | Web |
|---|---|---|---|---|
| `initBindings()` | Sync | ✅ | ✅ | ❌ |
| `initBindingsAsync()` | Async | ✅ | ✅ | ✅ |

Both functions accept an optional path argument to override the auto-detected
native library location. In most cases you don't need to pass a path.
