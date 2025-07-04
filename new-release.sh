#!/usr/bin/env bash

set -ex
old_version="1\.12\.3"
new_version="1\.12\.4"
replace_str="s/$old_version/$new_version/g"

sed -i.bak "$replace_str" ./sherpa-onnx/csrc/version.cc
sha1=$(git describe --match=NeVeRmAtCh --always --abbrev=8)
date=$(git log -1 --format=%ad --date=local)

sed -i.bak "s/  static const char \*sha1.*/  static const char \*sha1 = \"$sha1\";/g" ./sherpa-onnx/csrc/version.cc
sed -i.bak "s/  static const char \*date.*/  static const char \*date = \"$date\";/g" ./sherpa-onnx/csrc/version.cc

sed -i.bak "$replace_str" ./build-ios-shared.sh
sed -i.bak "$replace_str" ./pom.xml
sed -i.bak "$replace_str" ./jitpack.yml
sed -i.bak "$replace_str" ./android/SherpaOnnxAar/README.md

find android -name build.gradle -type f -exec sed -i.bak "s/sherpa-onnx:v$old_version/sherpa-onnx:v$new_version/g" {} \;

find flutter -name "*.yaml" -type f -exec sed -i.bak "$replace_str" {} \;
find dart-api-examples -name "*.yaml" -type f -exec sed -i.bak "$replace_str" {} \;
find flutter-examples -name "*.yaml" -type f -exec sed -i.bak "$replace_str" {} \;
find flutter -name "*.podspec" -type f -exec sed -i.bak "$replace_str" {} \;
find nodejs-addon-examples -name package.json -type f -exec sed -i.bak "$replace_str" {} \;
find nodejs-examples -name package.json -type f -exec sed -i.bak "$replace_str" {} \;

find harmony-os -name "README.md" -type f -exec sed -i.bak "$replace_str" {} \;
find harmony-os -name oh-package.json5 -type f -exec sed -i.bak "$replace_str" {} \;

find mfc-examples -name "README.md" -type f -exec sed -i.bak "$replace_str" {} \;

find . -name "*.bak" -exec rm {} \;
