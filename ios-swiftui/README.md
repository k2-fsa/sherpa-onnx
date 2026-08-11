# iOS SwiftUI Examples

## Overview

The sherpa-onnx Swift package provides two products for iOS:

- **`sherpa-onnx`** — uses a **static library** (`.a`) inside an xcframework. The sherpa-onnx code is compiled directly into your app binary at build time. This is the default and recommended option for most apps.
- **`sherpa-onnx-shared`** — uses a **shared library** (`.dylib` / `.framework`) inside an xcframework. The sherpa-onnx code is packaged as a separate dynamic framework and loaded at runtime. Your app bundle will contain an additional `.framework` file.

Both products expose the same Swift API. The only difference is how the native C++ code is linked.

| Product | Library Type | Swift Import |
|---|---|---|
| `sherpa-onnx` | Static library (`.a` in xcframework) | `import SherpaOnnx` |
| `sherpa-onnx-shared` | Shared library (`.dylib` in xcframework) | `import SherpaOnnxShared` |

## Using SPM (Swift Package Manager)

There are two ways to add sherpa-onnx to your Xcode project:

### Option A: Remote Package (from GitHub)

1. In Xcode, open your project
2. Go to **File → Add Package Dependencies...**
3. Enter the repository URL:
   ```
   https://github.com/k2-fsa/sherpa-onnx
   ```
4. Choose your version rule:
   - **Up to Next Major**: e.g. `1.13.0 ..< 2.0.0`
   - **Exact**: e.g. `1.13.5`
   - **Branch**: e.g. `master`
5. Click **Add Package**
6. Select the product you want (`sherpa-onnx` or `sherpa-onnx-shared`) and click **Add Package**

### Option B: Local Package (from source)

If you have the sherpa-onnx source code checked out locally:

1. In Xcode, open your project
2. Go to **File → Add Package Dependencies...**
3. Click **Add Local...** in the bottom-left corner
4. Navigate to the sherpa-onnx source directory and click **Add Package**
5. Select the product you want (`sherpa-onnx` or `sherpa-onnx-shared`) and click **Add Package**

Alternatively, you can manually edit the `.pbxproj` file:

```
/* XCLocalSwiftPackageReference section */
C96194D3301EED750025FCCA /* XCLocalSwiftPackageReference "../../path/to/sherpa-onnx" */ = {
    isa = XCLocalSwiftPackageReference;
    relativePath = "../../path/to/sherpa-onnx";
};
```

### Verifying the Package is Added

After adding the package, you should see it listed in:

- **Project Navigator** (left sidebar) → under the **Package Dependencies** section
- **Target → General → Frameworks, Libraries, and Embedded Content**

## Choosing Between Static and Shared Libraries

The sherpa-onnx package provides two library variants, each backed by its own xcframework:

| Product | XCFramework Type | Swift Import |
|---|---|---|
| `sherpa-onnx` | Static xcframework (default) | `import SherpaOnnx` |
| `sherpa-onnx-shared` | Shared/dynamic xcframework | `import SherpaOnnxShared` |

- **`sherpa-onnx`** links statically — the sherpa-onnx code is compiled directly into your app binary. This is the default and recommended option for most apps.
- **`sherpa-onnx-shared`** links dynamically — the sherpa-onnx code lives in a separate `.framework` that is embedded into your app bundle at runtime.

### How to Switch

1. **In Xcode**, open your project and go to the target's **General** tab
2. Under **Frameworks, Libraries, and Embedded Content**, remove the current library
3. Click **+** and add the desired product (`sherpa-onnx` or `sherpa-onnx-shared`)
4. Rebuild — no Swift code changes needed (see below)

### Auto-detecting the Import

You do **not** need to manually change your import statements when switching between `sherpa-onnx` and `sherpa-onnx-shared`. Use `#if canImport(...)` in your Swift code to automatically detect which product is available:

```swift
#if canImport(SherpaOnnx)
import SherpaOnnx
#elseif canImport(SherpaOnnxShared)
import SherpaOnnxShared
#else
#error("SherpaOnnx module not found. Please check your SPM dependency configuration.")
#endif
```

This way the same source code works with either product. The `#if canImport(...)` check runs at compile time — if `SherpaOnnx` is available it uses that, otherwise it falls back to `SherpaOnnxShared`. If neither is found, the build fails with a clear error message.

## How to Clean Xcode SPM Cache

### Option 1: Xcode Menu

`File → Packages → Reset Package Caches`

### Option 2: Command Line

```bash
# Clear SPM artifact cache
rm -rf ~/Library/Caches/org.swift.swiftpm

# Clear Xcode DerivedData
rm -rf ~/Library/Developer/Xcode/DerivedData
```

### Option 3: Clean Build Folder

In Xcode: `Product → Clean Build Folder` (Shift+Cmd+K)

## Troubleshooting

### Checksum Mismatch

If you see an error like:

```
checksum of downloaded artifact of binary target 'SherpaOnnxIOS' (...) does not match checksum specified by the manifest (...)
```

This means Xcode's SPM cache has a stale artifact. Fix:

1. `File → Packages → Reset Package Caches`
2. Restart Xcode
3. Rebuild

### Missing Package Product

If you see `Missing package product 'sherpa-onnx-shared'`:

1. Verify the local package path is correct in project settings
2. Reset package caches
3. Re-resolve packages: `File → Packages → Resolve Package Versions`

### No Such Module

If you see `No such module 'SherpaOnnxShared'` or `No such module 'SherpaOnnx'`:

- Make sure you have added one of the two products (`sherpa-onnx` or `sherpa-onnx-shared`) to your target
- If using `#if canImport(...)` as recommended above, the code will automatically pick the right module
- If importing manually, make sure the import matches the product you added

### Duplicate Output File (Shared Library Only)

If you see an error like:

```
duplicate output file '...SherpaOnnxC.framework' on task: Copy ...
```

This is a known Xcode issue with SPM dynamic (shared) frameworks — Xcode sometimes generates duplicate "Copy" build phases. Fix:

1. Close Xcode
2. Delete DerivedData:
   ```bash
   rm -rf ~/Library/Developer/Xcode/DerivedData/SherpaOnnxTts-*
   ```
3. Open Xcode
4. `File → Packages → Reset Package Caches`
5. Rebuild

If it still happens, open the target's **Build Phases** tab in Xcode and look for duplicate **Embed Frameworks** or **Copy Files** phases that both reference `SherpaOnnxC.framework`. Remove the duplicate if found.
