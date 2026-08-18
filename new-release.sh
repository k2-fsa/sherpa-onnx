#!/usr/bin/env bash

set -ex

cd "$(dirname "$0")"

old_version_code=20260810
new_version_code=20260818

old_version="1\.13\.5"
new_version="1\.13\.6"

replace_str="s/$old_version/$new_version/g"

sed -i.bak "$replace_str" ./CMakeLists.txt

sed -i.bak "$replace_str" ./sherpa-onnx/csrc/version.cc
sha1=$(git describe --match=NeVeRmAtCh --always --abbrev=8)
date=$(git log -1 --format=%ad --date=local)

find android -name "build.gradle" -type f -exec sed -i.bak "s/versionName \"$old_version\"/versionName \"$new_version\"/g" {} \;
find android -name "build.gradle.kts" -type f -exec sed -i.bak "s/versionName = \"$old_version\"/versionName = \"$new_version\"/g" {} \;

find android -name "build.gradle" -type f -exec sed -i.bak "s/versionCode $old_version_code/versionCode $new_version_code/g" {} \;
find android -name "build.gradle.kts" -type f -exec sed -i.bak "s/versionCode = $old_version_code/versionCode = $new_version_code/g" {} \;

find . -name "build*.sh" -type f -exec sed -i.bak "s/$old_version_code/$new_version_code/g" {} \;

sed -i.bak "s/  static const char \*sha1.*/  static const char \*sha1 = \"$sha1\";/g" ./sherpa-onnx/csrc/version.cc
sed -i.bak "s/  static const char \*date.*/  static const char \*date = \"$date\";/g" ./sherpa-onnx/csrc/version.cc

find scripts/wheel -name "setup.py" -type f -exec sed -i.bak "$replace_str" {} \;
sed -i.bak "$replace_str" ./setup.py

find ./ios-swift -name "project.pbxproj" -type f -exec sed -i.bak "$replace_str" {} \;
find ./ios-swiftui -name "project.pbxproj" -type f -exec sed -i.bak "$replace_str" {} \;
find ./ios-swiftui -name "README.md" -type f -exec sed -i.bak "$replace_str" {} \;
find ./spm-examples -name "Package.swift" -type f -exec sed -i.bak "$replace_str" {} \;

sed -i.bak "$replace_str" ./build-ios.sh
sed -i.bak "$replace_str" ./build-ios-shared.sh
sed -i.bak "$replace_str" ./build-ios-no-tts.sh
sed -i.bak "$replace_str" ./build-macos.sh
sed -i.bak "$replace_str" ./build-macos-shared.sh
sed -i.bak "$replace_str" ./build-macos-shared-sherpa-with-static-onnxruntime.sh
sed -i.bak "$replace_str" ./build-ios-shared-sherpa-with-static-onnxruntime.sh

for f in ./build*.sh; do
  sed -i.bak "s/$old_version_code/$new_version_code/g" "$f"
done

sed -i.bak "$replace_str" ./pom.xml
sed -i.bak "$replace_str" ./sherpa-onnx/java-api/pom.xml
sed -i.bak "$replace_str" ./sherpa-onnx/java-api/README.md
sed -i.bak "$replace_str" ./jitpack.yml
sed -i.bak "$replace_str" ./android/SherpaOnnxAar/README.md

find java-api-examples -name "README.md" -type f -exec sed -i.bak "$replace_str" {} \;
find java-api-examples -name "build.gradle" -type f -exec sed -i.bak "$replace_str" {} \;
find java-api-examples -name "build.gradle.kts" -type f -exec sed -i.bak "$replace_str" {} \;
find java-api-examples -name "pom.xml" -type f -exec sed -i.bak "$replace_str" {} \;

sed -i.bak "$replace_str" ./rust-api-examples/Cargo.toml
sed -i.bak "$replace_str" ./rust-api-examples/for-advanced-users.md
sed -i.bak "$replace_str" ./rust-api-examples/README.md
sed -i.bak "$replace_str" ./sherpa-onnx/rust/sherpa-onnx-sys/Cargo.toml
sed -i.bak "$replace_str" ./sherpa-onnx/rust/sherpa-onnx/Cargo.toml
sed -i.bak "$replace_str" ./sherpa-onnx/rust/sherpa-onnx/src/lib.rs
sed -i.bak "$replace_str" ./sherpa-onnx/rust/sherpa-onnx/README.md

sed -i.bak "$replace_str" ./tauri-examples/non-streaming-speech-recognition-from-file/package.json
sed -i.bak "$replace_str" ./tauri-examples/non-streaming-speech-recognition-from-file/src-tauri/Cargo.toml
sed -i.bak "$replace_str" ./tauri-examples/non-streaming-speech-recognition-from-file/src-tauri/tauri.conf.json

sed -i.bak "$replace_str" ./tauri-examples/non-streaming-speech-recognition-from-microphone/package.json
sed -i.bak "$replace_str" ./tauri-examples/non-streaming-speech-recognition-from-microphone/src-tauri/Cargo.toml
sed -i.bak "$replace_str" ./tauri-examples/non-streaming-speech-recognition-from-microphone/src-tauri/tauri.conf.json

find android -name build.gradle -type f -exec sed -i.bak "s/sherpa-onnx:v$old_version/sherpa-onnx:v$new_version/g" {} \;
find android -name build.gradle.kts -type f -exec sed -i.bak "s/sherpa-onnx:v$old_version/sherpa-onnx:v$new_version/g" {} \;

find flutter -name "*.yaml" -type f -exec sed -i.bak "$replace_str" {} \;
find dart-api-examples -name "*.yaml" -type f -exec sed -i.bak "$replace_str" {} \;
find dart-api-examples -name "*.md" -type f -exec sed -i.bak "$replace_str" {} \;
find flutter-examples -name "*.yaml" -type f -exec sed -i.bak "$replace_str" {} \;
find flutter -name "*.podspec" -type f -exec sed -i.bak "$replace_str" {} \;
find nodejs-addon-examples -name package.json -type f -exec sed -i.bak "$replace_str" {} \;
find nodejs-examples -name package.json -type f -exec sed -i.bak "$replace_str" {} \;

find harmony-os -name "README.md" -type f -exec sed -i.bak "$replace_str" {} \;
find harmony-os -name oh-package.json5 -type f -exec sed -i.bak "$replace_str" {} \;
find harmony-os -name BuildProfile.ets -type f -exec sed -i.bak "$replace_str" {} \;

find mfc-examples -name "README.md" -type f -exec sed -i.bak "$replace_str" {} \;

find .github/workflows -name "*.yaml" -type f -exec sed -i.bak "$replace_str" {} \;

find . -name "*.bak" -exec rm {} \;

