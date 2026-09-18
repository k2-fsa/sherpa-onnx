# Reproducible F-Droid builds of the Android library (AAR)

This page documents how to build the sherpa-onnx Android library (AAR) so that
an [F-Droid](https://f-droid.org) `srclib` recipe produces APKs that are
byte-identical to a developer's reference build. F-Droid builds apps from
source, so a prebuilt AAR dependency must be replaced by an srclib entry
pinned to a sherpa-onnx commit.

Two distinct checks are involved, and it pays to keep them apart:

- `AllowedAPKSigningKeys` in the fdroiddata recipe pins which signing key F-Droid
  accepts. It says nothing about the build itself.
- Reproducibility is proven by fdroidserver's verification: the buildserver
  rebuilds from source and compares the result byte-for-byte against the
  developer's reference APK (see the verification commands at the end).

Everything below was worked out while packaging
[Anti-Vocale](https://github.com/RisorseArtificiali/anti-vocale), whose F-Droid
releases verify byte-identical against the reference builds. The working
references are the fdroiddata recipe for `com.antivocale.app` and the app's
[release runbook](https://github.com/RisorseArtificiali/anti-vocale/blob/main/docs/release-runbook.md).

## 1. Pin the commit everywhere, once

The app's fetch script, the fdroiddata srclib block, and any version-marker
file in the app repo must all reference the same sherpa-onnx commit. Desync
between these three is the most common cause of a verification failure.

## 2. Build flags

Configure with `SHERPA_ONNX_ENABLE_C_API=OFF` and `SHERPA_ONNX_ENABLE_JNI=ON`.
With the C API enabled the AAR ships extra `libsherpa-onnx-c-api.so` and
`libsherpa-onnx-cxx-api.so` libraries that a JNI-only consumer never loads, and
they break the byte comparison.

## 3. Pin the exact NDK revision

The packaged `.so` files are stripped with the NDK's `llvm-strip`, and patch
releases of the same NDK major ship different LLVM revisions (r27b, r27c and
r27d all differ), so "the same major" is not enough. Pin the full revision on
both sides: `ndkVersion = "27.0.12077973"` in the consuming app's
`build.gradle.kts` (or the equivalent), and the same exact value in the
recipe's `ndk:` field so fdroidserver installs that revision.

## 4. Disable baseline-profile generation in the consuming project

AGP's ArtProfile tasks write `assets/dexopt/baseline.prof` and `baseline.profm`
with non-deterministic content (per-build ordering variation), which breaks the
byte comparison for a runtime-only optimization. Disable the tasks explicitly:

```kotlin
tasks.whenTaskAdded {
    if (name.contains("ArtProfile")) {
        enabled = false
    }
}
```

Use `whenTaskAdded`, not `afterEvaluate`: the ArtProfile tasks are created
lazily after project evaluation, so `afterEvaluate` matches zero of them.

## 5. Per-ABI APKs

The library AAR itself can stay fat (all ABIs in one AAR). Per-ABI artifacts
come from the consuming app's APK splits, and AGP strips the unused library
ABIs from each split APK:

```kotlin
splits {
    abi {
        isEnable = true
        reset()
        include("arm64-v8a", "armeabi-v7a", "x86_64")
        isUniversalApk = false
    }
}
```

Give each ABI a distinct versionCode (for example base*10 plus 1/2/4 for
armeabi-v7a/arm64-v8a/x86_64) so F-Droid serves the correct APK per device.

## 6. Verification tooling

With points 1 to 5 in place, compare the rebuilt APK against the signed
reference with
[reproducible-apk-tools](https://github.com/obfusk/reproducible-apk-tools):

```bash
# align the rebuilt unsigned APK the way apksigner pads (page size 16)
zipalign.py -p 16 --pad-like-apksigner rebuilt-unsigned.apk

# copy the reference APK's signature block onto the rebuilt APK
apksigcopier copy --v1-only reference-signed.apk rebuilt-unsigned.apk rebuilt-signed.apk

# byte comparison of the two
apksigcopier compare reference-signed.apk rebuilt-signed.apk
```

See each tool's README for the full flag set; the combination above is what
the Anti-Vocale verification uses.

## Reference recipe

The srclib block of `com.antivocale.app` in
[fdroiddata](https://gitlab.com/fdroid/fdroiddata) builds the library from
source with the flags above and is reviewed by the F-Droid team; the same
pattern applies to any app consuming sherpa-onnx on F-Droid.
