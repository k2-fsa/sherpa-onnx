#!/usr/bin/env bash

./build-android-arm64-v8a.sh
cp build-android-arm64-v8a/install/lib/*.so /Users/iprovalov/github/lingofonex-android/app/src/main/jniLibs/arm64-v8a/
cp build-android-arm64-v8a/install/lib/*.so /Users/iprovalov/github/lingofonex-android/app/src/main/jniLibsUnstripped/arm64-v8a/
$ANDROID_NDK/toolchains/llvm/prebuilt/darwin-x86_64/bin/llvm-strip --strip-all /Users/iprovalov/github/lingofonex-android/app/src/main/jniLibs/arm64-v8a/*.so

./build-android-armv7-eabi.sh
cp build-android-armv7-eabi/install/lib/*.so /Users/iprovalov/github/lingofonex-android/app/src/main/jniLibs/armeabi-v7a/
cp build-android-armv7-eabi/install/lib/*.so /Users/iprovalov/github/lingofonex-android/app/src/main/jniLibsUnstripped/armeabi-v7a/
$ANDROID_NDK/toolchains/llvm/prebuilt/darwin-x86_64/bin/llvm-strip --strip-all /Users/iprovalov/github/lingofonex-android/app/src/main/jniLibs/armeabi-v7a/*.so

./build-android-x86.sh
cp build-android-x86/install/lib/*.so /Users/iprovalov/github/lingofonex-android/app/src/main/jniLibs/x86/
cp build-android-x86/install/lib/*.so /Users/iprovalov/github/lingofonex-android/app/src/main/jniLibsUnstripped/x86/
$ANDROID_NDK/toolchains/llvm/prebuilt/darwin-x86_64/bin/llvm-strip --strip-all /Users/iprovalov/github/lingofonex-android/app/src/main/jniLibs/x86/*.so

./build-android-x86-64.sh
cp build-android-x86-64/install/lib/*.so /Users/iprovalov/github/lingofonex-android/app/src/main/jniLibs/x86_64/
cp build-android-x86-64/install/lib/*.so /Users/iprovalov/github/lingofonex-android/app/src/main/jniLibsUnstripped/x86_64/
$ANDROID_NDK/toolchains/llvm/prebuilt/darwin-x86_64/bin/llvm-strip --strip-all /Users/iprovalov/github/lingofonex-android/app/src/main/jniLibs/x86_64/*.so
