#!/usr/bin/env bash

set -ex

sed -i.bak 's/1\.12\.0/1\.12\.1/g' ./sherpa-onnx/csrc/version.cc
sha1=$(git describe --match=NeVeRmAtCh --always --abbrev=8)
date=$(git log -1 --format=%ad --date=local)

sed -i.bak "s/  static const char \*sha1.*/  static const char \*sha1 = \"$sha1\";/g" ./sherpa-onnx/csrc/version.cc
sed -i.bak "s/  static const char \*date.*/  static const char \*date = \"$date\";/g" ./sherpa-onnx/csrc/version.cc

sed -i.bak 's/1\.12\.0/1\.12\.1/g' ./build-ios-shared.sh
sed -i.bak 's/1\.12\.0/1\.12\.1/g' ./pom.xml
sed -i.bak 's/1\.12\.0/1\.12\.1/g' ./jitpack.yml
sed -i.bak 's/1\.12\.0/1\.12\.1/g' ./android/SherpaOnnxAar/README.md

find android -name build.gradle -type f -exec sed -i.bak 's/sherpa-onnx:v1\.12\.0/sherpa-onnx:v1\.12\.1/g' {} \;

find flutter -name *.yaml -type f -exec sed -i.bak 's/1\.12\.0/1\.12\.1/g' {} \;
find dart-api-examples -name *.yaml -type f -exec sed -i.bak 's/1\.12\.0/1\.12\.1/g' {} \;
find flutter-examples -name *.yaml -type f -exec sed -i.bak 's/1\.12\.0/1\.12\.1/g' {} \;
find flutter -name *.podspec -type f -exec sed -i.bak 's/1\.12\.0/1\.12\.1/g' {} \;
find nodejs-addon-examples -name package.json -type f -exec sed -i.bak 's/1\.12\.0/1\.12\.1/g' {} \;
find nodejs-examples -name package.json -type f -exec sed -i.bak 's/1\.12\.0/1\.12\.1/g' {} \;

find harmony-os -name "README.md" -type f -exec sed -i.bak 's/1\.12\.0/1\.12\.1/g' {} \;
find harmony-os -name oh-package.json5 -type f -exec sed -i.bak 's/1\.12\.0/1\.12\.1/g' {} \;

find mfc-examples -name "README.md" -type f -exec sed -i.bak 's/1\.12\.0/1\.12\.1/g' {} \;

find . -name "*.bak" -exec rm {} \;
