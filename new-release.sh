#!/usr/bin/env bash

set -ex

sed -i.bak 's/1\.10\.34/1\.10\.35/g' ./build-ios-shared.sh
sed -i.bak 's/1\.10\.34/1\.10\.35/g' ./pom.xml
sed -i.bak 's/1\.10\.34/1\.10\.35/g' ./jitpack.yml
sed -i.bak 's/1\.10\.34/1\.10\.35/g' ./android/SherpaOnnxAar/README.md

find flutter -name *.yaml -type f -exec sed -i.bak 's/1\.10\.34/1\.10\.35/g' {} \;
find dart-api-examples -name *.yaml -type f -exec sed -i.bak 's/1\.10\.34/1\.10\.35/g' {} \;
find flutter-examples -name *.yaml -type f -exec sed -i.bak 's/1\.10\.34/1\.10\.35/g' {} \;
find flutter -name *.podspec -type f -exec sed -i.bak 's/1\.10\.34/1\.10\.35/g' {} \;
find nodejs-addon-examples -name package.json -type f -exec sed -i.bak 's/1\.10\.34/1\.10\.35/g' {} \;

find harmony-os -name "README.md" -type f -exec sed -i.bak 's/1\.10\.34/1\.10\.35/g' {} \;
find harmony-os -name oh-package.json5 -type f -exec sed -i.bak 's/1\.10\.34/1\.10\.35/g' {} \;
