#!/usr/bin/env bash
#
# Test one xcframework scenario against the artifacts produced by
# .github/workflows/build-xcframework.yaml.
#
# Usage: test-xcframework.sh <scenario>
#
# Scenarios:
#   spm-macos-static   spm-ios-static
#   spm-macos-shared   spm-ios-shared
#   ios-apps-static    ios-apps-shared
#
# Each scenario is self-contained: it downloads its own artifacts and patches
# Package.swift itself, so the scenarios can run as separate parallel jobs.
# Previously they shared one job and depended on each other's edits.

set -euo pipefail

scenario=${1:?usage: $0 <spm-macos-static|spm-ios-static|spm-macos-shared|spm-ios-shared|ios-apps-static|ios-apps-shared>}

cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

# The 4 artifacts, unzipped where the root Package.swift expects them.
prepare_xcframeworks() {
  mkdir -p build-macos build-ios build-macos-shared build-ios-shared

  unzip -o artifacts/sherpa-onnx-macos-static-xcframework/*.zip -d build-macos/
  unzip -o artifacts/sherpa-onnx-ios-static-xcframework/*.zip -d build-ios/
  unzip -o artifacts/sherpa-onnx-macos-shared-xcframework/*.zip -d build-macos-shared/
  unzip -o artifacts/sherpa-onnx-ios-shared-xcframework/*.zip -d build-ios-shared/

  echo "=== macOS static xcframework ==="
  ls -lh build-macos/sherpa-onnx.xcframework/
  echo "=== iOS static xcframework ==="
  ls -lh build-ios/sherpa-onnx.xcframework/
  echo "=== macOS shared xcframework ==="
  ls -lh build-macos-shared/sherpa-onnx.xcframework/
  echo "=== iOS shared xcframework ==="
  ls -lh build-ios-shared/sherpa-onnx.xcframework/
}

# Point the root Package.swift binary targets at the extracted xcframeworks
# instead of the released URLs.
patch_root_package_to_local() {
  python3 - <<'PY'
import re

paths = {
    "SherpaOnnxMacOS": "build-macos/sherpa-onnx.xcframework",
    "SherpaOnnxIOS": "build-ios/sherpa-onnx.xcframework",
    "SherpaOnnxMacOSShared": "build-macos-shared/sherpa-onnx.xcframework",
    "SherpaOnnxIOSShared": "build-ios-shared/sherpa-onnx.xcframework",
}
text = open("Package.swift").read()
for name, path in paths.items():
    text, n = re.subn(
        r'\.binaryTarget\(\s*name: "%s",\s*url:\s*"[^"]+",\s*checksum: "[a-f0-9]+"\s*\)' % name,
        '.binaryTarget(name: "%s", path: "%s")' % (name, path),
        text,
    )
    if n != 1:
        raise SystemExit("failed to patch binaryTarget %s (matched %d times)" % (name, n))
open("Package.swift", "w").write(text)
print("patched root Package.swift binary targets to local paths")
PY
}

# Point an SPM example at the repo root package instead of the published one.
use_local_package() {
  sed -i.bak 's|.package(url: "https://github.com/k2-fsa/sherpa-onnx", exact: "2.0.1")|.package(path: "../../")|' "$1"
}

use_shared_product() {
  sed -i.bak2 's|.product(name: "sherpa-onnx", package: "sherpa-onnx")|.product(name: "sherpa-onnx-shared", package: "sherpa-onnx")|' "$1"
}

# Boot an iPhone simulator and print only its UDID on stdout. Everything else
# goes to stderr so `DEVICE=$(boot_simulator)` captures just the UDID.
boot_simulator() {
  local device
  device=$(xcrun simctl list devices available -j | python3 -c "
import json, sys
devices = json.load(sys.stdin)['devices']
for key in devices:
    for d in devices[key]:
        if d['isAvailable'] and 'iPhone' in d['name']:
            print(d['udid'])
            sys.exit(0)
")
  echo "Device: $device" >&2
  xcrun simctl boot "$device" >/dev/null 2>&1 || true
  echo "$device"
}

build_ios_sim_example() {
  xcodebuild -scheme SherpaOnnxExample \
    -destination "generic/platform=iOS Simulator" \
    -skipPackagePluginValidation \
    -verbose \
    build 2>&1

  echo "=== Finding built executable ==="
  BINARY=$(find ~/Library/Developer/Xcode/DerivedData -path "*/Build/Products/*-iphonesimulator/SherpaOnnxExample" -type f | head -1)
  echo "Binary: $BINARY"
}

spm_macos_static() {
  use_local_package spm-examples/macOS/Package.swift

  echo "=== Package.swift ==="
  cat spm-examples/macOS/Package.swift
  echo "=== Root Package.swift ==="
  cat Package.swift

  cd spm-examples/macOS
  rm -rf .build
  swift build --verbose
  swift run SherpaOnnxExample
}

spm_macos_shared() {
  use_local_package spm-examples/macOS/Package.swift
  use_shared_product spm-examples/macOS/Package.swift

  echo "=== Package.swift (shared) ==="
  cat spm-examples/macOS/Package.swift
  echo "=== Root Package.swift ==="
  cat Package.swift

  cd spm-examples/macOS
  rm -rf .build
  swift build --verbose

  # Find the shared onnxruntime dylib from OnnxruntimeMacOSShared artifact
  ort_dylib=$(find .build/artifacts -path "*OnnxruntimeMacOSShared*" -name "onnxruntime" -type f | grep -v Headers | head -1)
  if [ -z "$ort_dylib" ]; then
    echo "ERROR: shared onnxruntime dylib not found in artifacts"
    echo "=== Listing all artifacts ==="
    find .build/artifacts -type f | sort
    exit 1
  fi
  echo "Found onnxruntime dylib: $ort_dylib"
  file "$ort_dylib"

  exe_dir=$(dirname "$(find .build -name "SherpaOnnxExample" -type f -perm +111 | head -1)")
  cp "$ort_dylib" "$exe_dir/libonnxruntime.dylib"
  install_name_tool -id "@rpath/libonnxruntime.dylib" "$exe_dir/libonnxruntime.dylib"

  swift run SherpaOnnxExample
}

spm_ios_static() {
  use_local_package spm-examples/iOS/Package.swift

  echo "=== Package.swift ==="
  cat spm-examples/iOS/Package.swift

  cd spm-examples/iOS
  echo "=== Resolving ==="
  swift package resolve

  echo "=== Building for iOS Simulator ==="
  build_ios_sim_example

  echo "=== Booting iOS Simulator ==="
  DEVICE=$(boot_simulator)

  echo "=== Running on iOS Simulator ==="
  xcrun simctl spawn "$DEVICE" "$BINARY"
}

spm_ios_shared() {
  use_local_package spm-examples/iOS/Package.swift
  use_shared_product spm-examples/iOS/Package.swift

  echo "=== Package.swift (shared) ==="
  cat spm-examples/iOS/Package.swift

  cd spm-examples/iOS
  echo "=== Resolving ==="
  swift package resolve

  echo "=== Building for iOS Simulator ==="
  build_ios_sim_example

  PRODUCTS_DIR=$(dirname "$BINARY")

  # Ensure dynamic frameworks are accessible next to the binary
  # simctl spawn uses @executable_path-relative rpaths, so frameworks must be
  # in PackageFrameworks/ or Frameworks/ relative to the binary
  echo "=== Checking framework locations ==="
  find "$PRODUCTS_DIR" -name "*.framework" -type d
  ls -la "$PRODUCTS_DIR/Frameworks/" 2>/dev/null || true
  ls -la "$PRODUCTS_DIR/PackageFrameworks/" 2>/dev/null || true

  # Copy SherpaOnnxC.framework to PackageFrameworks/ if not already there
  pkg_fw_dir="$PRODUCTS_DIR/PackageFrameworks"
  mkdir -p "$pkg_fw_dir"

  sherpa_fw=$(find "$PRODUCTS_DIR" -path "*/Frameworks/SherpaOnnxC.framework" -type d | head -1)
  if [ -n "$sherpa_fw" ] && [ ! -d "$pkg_fw_dir/SherpaOnnxC.framework" ]; then
    cp -a "$sherpa_fw" "$pkg_fw_dir/"
  fi

  ort_fw=$(find "$PRODUCTS_DIR" -path "*/Frameworks/onnxruntime.framework" -type d | head -1)
  if [ -n "$ort_fw" ] && [ ! -d "$pkg_fw_dir/onnxruntime.framework" ]; then
    cp -a "$ort_fw" "$pkg_fw_dir/"
  fi

  echo "=== PackageFrameworks contents ==="
  ls -la "$pkg_fw_dir/"

  echo "=== Booting iOS Simulator ==="
  DEVICE=$(boot_simulator)

  echo "=== Running on iOS Simulator ==="
  xcrun simctl spawn "$DEVICE" "$BINARY"
}

# Patch the app projects to consume the repo-root package rather than the
# published one.
use_local_package_in_ios_apps() {
  find ios-swiftui ios-swift -name "project.pbxproj" -exec sed -i '' \
    -e '/\/\* XCRemoteSwiftPackageReference "sherpa-onnx" \*\/ = {/,/^\t\t};/{s|isa = XCRemoteSwiftPackageReference|isa = XCLocalSwiftPackageReference|;s|repositoryURL = "https://github.com/k2-fsa/sherpa-onnx"|relativePath = ../../|;/requirement = {/,/};/d;}' \
    {} +
}

build_ios_apps() {
  for app in SherpaOnnxAsr SherpaOnnx2Pass SherpaOnnxTtsLexicon SherpaOnnxTtsPhonemize; do
    echo "=== Building ios-swiftui/$app ==="
    (
      cd ios-swiftui/$app
      xcodebuild -project $app.xcodeproj \
        -scheme $app \
        -destination 'generic/platform=iOS Simulator' \
        -skipPackagePluginValidation \
        build
    )
  done

  echo "=== Building ios-swift/SherpaOnnxAsr ==="
  (
    cd ios-swift/SherpaOnnxAsr
    xcodebuild -project SherpaOnnxAsr.xcodeproj \
      -scheme SherpaOnnxAsr \
      -destination 'generic/platform=iOS Simulator' \
      -skipPackagePluginValidation \
      build
  )
}

ios_apps_static() {
  use_local_package_in_ios_apps
  build_ios_apps
}

ios_apps_shared() {
  # Swap static xcframeworks with shared ones in the local paths
  rm -rf build-macos/sherpa-onnx.xcframework
  cp -a build-macos-shared/sherpa-onnx.xcframework build-macos/
  rm -rf build-ios/sherpa-onnx.xcframework
  cp -a build-ios-shared/sherpa-onnx.xcframework build-ios/

  # Patch root Package.swift to use shared onnxruntime products
  python3 -c "
text = open('Package.swift').read()
# In the static wrapper target, swap to shared onnxruntime products
text = text.replace('\"onnxruntime-macos\"', '\"onnxruntime-macos-shared\"')
text = text.replace('\"onnxruntime-ios\"', '\"onnxruntime-ios-shared\"')
open('Package.swift', 'w').write(text)
"

  use_local_package_in_ios_apps
  build_ios_apps
}

echo "================================================================"
echo "xcframework scenario: $scenario"
echo "================================================================"

prepare_xcframeworks
patch_root_package_to_local

case "$scenario" in
  spm-macos-static) spm_macos_static ;;
  spm-macos-shared) spm_macos_shared ;;
  spm-ios-static) spm_ios_static ;;
  spm-ios-shared) spm_ios_shared ;;
  ios-apps-static) ios_apps_static ;;
  ios-apps-shared) ios_apps_shared ;;
  *) echo "unknown scenario: $scenario"; exit 1 ;;
esac

echo "================================================================"
echo "scenario passed: $scenario"
echo "================================================================"
