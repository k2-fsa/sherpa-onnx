# Tauri hello world

This is a minimal Tauri app that displays sherpa-onnx version information:

- sherpa-onnx version
- Git SHA1
- Git date
- Onnxruntime version

It supports **Linux, macOS, Windows, iOS, and Android**.

## Prerequisites

- [Rust](https://www.rust-lang.org/tools/install) (stable)
- [Node.js](https://nodejs.org/) (for the Tauri CLI)
- Tauri CLI: `cargo install tauri-cli`
- Platform-specific Tauri prerequisites: see [Tauri prerequisites guide](https://v2.tauri.app/start/prerequisites/)

## Build from source (using local sherpa-onnx)

```bash
# From the repository root
cd tauri-examples/hello_world

# Use local sherpa-onnx crate (optional, for development)
mkdir -p src-tauri/.cargo
cat > src-tauri/.cargo/config.toml <<'EOF'
[patch.crates-io]
sherpa-onnx = { path = "../../../sherpa-onnx/rust/sherpa-onnx" }
sherpa-onnx-sys = { path = "../../../sherpa-onnx/rust/sherpa-onnx-sys" }
EOF

npm install
npm run build
```

The built app will be in `src-tauri/target/release/bundle/`.

## Run in development mode

```bash
cd tauri-examples/hello_world
npm install
npm run dev
```

## Running a downloaded app on macOS

If you download the `.zip` from GitHub Releases and see:

> "hello_world" is damaged and can't be opened. You should move it to the Trash.

This is macOS Gatekeeper blocking an unsigned app. Fix it with:

```bash
xattr -cr hello_world.app
```

Then double-click `hello_world.app` to run it.

## Build for specific platforms

### Desktop (Linux, macOS, Windows)

```bash
npm run build
```

### iOS

iOS uses **shared linking only**. For the very first build, run the setup
script to download the xcframework:

```bash
cd tauri-examples/hello_world
src-tauri/setup-ios.sh
cargo tauri ios init
cargo tauri ios build --target aarch64 --no-sign
```

After the first build, `build.rs` caches the xcframework and subsequent
builds are automatic.

#### Build for a real device

```bash
cd tauri-examples/hello_world
npm install

# Generate the Xcode project (one-time)
cargo tauri ios init

# Build unsigned IPA for real device
cargo tauri ios build --target aarch64 --no-sign
```

Output: `src-tauri/gen/apple/build/arm64/hello_world.ipa`

#### Build for simulator

```bash
cargo tauri ios build --target aarch64-sim --no-sign
```

Output: look for `hello_world.app` in `src-tauri/gen/apple/`

#### Run on simulator (using pre-built app)

Download `tauri-hello-world-ios-simulator.zip` from
[tauri releases](https://github.com/k2-fsa/sherpa-onnx/releases/tag/tauri),
then:

```bash
# Unzip
unzip tauri-hello-world-ios-simulator.zip

# Boot a simulator
xcrun simctl boot "iPhone 16"

# Open Simulator app
open -a Simulator

# Install and launch
xcrun simctl install "iPhone 16" hello_world.app
xcrun simctl launch "iPhone 16" com.k2fsa.sherpa.onnx.hello.world
```

#### Run on simulator (building from source)

```bash
# List available simulators
xcrun simctl list devices | grep iPhone

# Boot a simulator
xcrun simctl boot "iPhone 16"

# Install the .app (from simulator build)
xcrun simctl install "iPhone 16" \
  src-tauri/gen/apple/build/arm64-sim/hello_world.app

# Launch
xcrun simctl launch "iPhone 16" com.k2fsa.sherpa.onnx.hello.world

# Open Simulator app to see it
open -a Simulator
```

Or open the Xcode project and press **⌘R** with a simulator selected:

```bash
open src-tauri/gen/apple/hello_world.xcodeproj
```

#### Run on a real device

The IPA from `cargo tauri ios build --no-sign` is unsigned. Re-sign it first:

```bash
codesign --force --sign "iPhone Developer: Your Name (TEAMID)" \
  --entitlements entitlements.plist \
  hello_world.ipa
```

Then install via Xcode (Window → Devices and Simulators) or `ios-deploy`:

```bash
brew install ios-deploy
ios-deploy --bundle hello_world.ipa
```

See [README-iOS.md](../README-iOS.md) for the full iOS guide.

### Android

`build.rs` downloads the pre-built `.so` libraries automatically and copies them
into the Tauri Android project's `jniLibs` directory so Gradle bundles them into
the APK.

#### Build

```bash
cd tauri-examples/hello_world
npm install

# Generate the Android project (one-time)
cargo tauri android init

# Build a debug APK (build.rs downloads .so files here)
cargo tauri android build --apk --debug
```

The APK is generated at
`src-tauri/gen/android/app/build/outputs/apk/universal/debug/app-universal-debug.apk`.

#### Install on a connected device or emulator

```bash
# List connected devices
adb devices

# Install the APK
adb install src-tauri/gen/android/app/build/outputs/apk/universal/debug/app-universal-debug.apk

# Launch the app
adb shell am start -n com.k2fsa.sherpa.onnx.hello.world/.MainActivity
```

#### Build for a specific architecture

```bash
# arm64 (most modern phones)
cargo tauri android build --target aarch64 --apk --debug

# armv7 (older phones)
cargo tauri android build --target armv7 --apk --debug

# x86_64 (emulators)
cargo tauri android build --target x86_64 --apk --debug
```

#### Run on an emulator

```bash
# List available emulators
emulator -list-avds

# Start an emulator
emulator -avd Pixel_6_API_34 &

# Then build and install
cargo tauri android build --target x86_64 --apk --debug
adb install path/to/app-universal-debug.apk
```
