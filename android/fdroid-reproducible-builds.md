# Reproducible F-Droid builds of the Android library (AAR)

This page documents how to build the sherpa-onnx Android library (AAR) so that
an [F-Droid](https://f-droid.org) `srclib` recipe produces APKs that are
byte-identical to a developer's reference build. F-Droid builds apps from
source, so a prebuilt AAR dependency must be replaced by an srclib entry
pinned to a sherpa-onnx commit, and F-Droid verifies reproducibility with
`AllowedAPKSigningKeys`.

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

## 3. NDK version and stripping

Pin the NDK version: the packaged `.so` files are stripped with the NDK's
`llvm-strip`, and different NDK versions produce different bytes. Set
`ndkVersion` in the consuming project's `build.gradle.kts` (or the equivalent
in the recipe) to match the toolchain you build with. sherpa-onnx releases are
built against NDK r27; using the same major keeps the strip output identical.

## 4. Disable baseline-profile generation in the consuming project

The Android Gradle plugin's ArtProfile task writes `baseline.prof` and
`baseline.profm` whose entry ordering varies between builds. For a library
consumer these files bring no functional benefit, so disable the generation;
otherwise the APK comparison fails on profile bytes alone.

## 5. Per-ABI artifacts

Build the AAR for each ABI your recipe ships (the fdroiddata recipes use
separate per-ABI versionCodes). A fat AAR makes per-ABI APKs differ in unused
library bytes even when the effective code is identical.

## 6. Verification tooling

Compare the final APKs with
[reproducible-apk-tools](https://github.com/obfusk/reproducible-apk-tools):
align with `zipalign --page-size 16 --pad-like-apksigner`, copy the signature
block with `apksigcopier`, then diff. On the library side nothing special is
needed once points 1 to 5 hold.

## Reference recipe

The srclib block of `com.antivocale.app` in
[fdroiddata](https://gitlab.com/fdroid/fdroiddata) builds the library from
source with the flags above and is reviewed by the F-Droid team; the same
pattern applies to any app consuming sherpa-onnx on F-Droid.
