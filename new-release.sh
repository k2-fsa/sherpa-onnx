#!/usr/bin/env bash

set -ex

sed -i.bak 's/1\.10\.39/1\.10\.40/g' ./build-ios-shared.sh
sed -i.bak 's/1\.10\.39/1\.10\.40/g' ./pom.xml
sed -i.bak 's/1\.10\.39/1\.10\.40/g' ./jitpack.yml
sed -i.bak 's/1\.10\.39/1\.10\.40/g' ./android/SherpaOnnxAar/README.md

find android -name build.gradle -type f -exec sed -i.bak 's/sherpa-onnx:v1\.10\.39/sherpa-onnx:v1\.10\.40/g' {} \;

find flutter -name *.yaml -type f -exec sed -i.bak 's/1\.10\.39/1\.10\.40/g' {} \;
find dart-api-examples -name *.yaml -type f -exec sed -i.bak 's/1\.10\.39/1\.10\.40/g' {} \;
find flutter-examples -name *.yaml -type f -exec sed -i.bak 's/1\.10\.39/1\.10\.40/g' {} \;
find flutter -name *.podspec -type f -exec sed -i.bak 's/1\.10\.39/1\.10\.40/g' {} \;
find nodejs-addon-examples -name package.json -type f -exec sed -i.bak 's/1\.10\.39/1\.10\.40/g' {} \;

find harmony-os -name "README.md" -type f -exec sed -i.bak 's/1\.10\.39/1\.10\.40/g' {} \;
find harmony-os -name oh-package.json5 -type f -exec sed -i.bak 's/1\.10\.39/1\.10\.40/g' {} \;
