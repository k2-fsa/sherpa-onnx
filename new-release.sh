#!/usr/bin/env bash

set -ex

sed -i.bak 's/1\.10\.33/1\.10\.34/g' ./build-ios-shared.sh
sed -i.bak 's/1\.10\.33/1\.10\.34/g' ./pom.xml

find flutter -name *.yaml -type f -exec sed -i.bak 's/1\.10\.33/1\.10\.34/g' {} \;
find dart-api-examples -name *.yaml -type f -exec sed -i.bak 's/1\.10\.33/1\.10\.34/g' {} \;
find flutter-examples -name *.yaml -type f -exec sed -i.bak 's/1\.10\.33/1\.10\.34/g' {} \;
find flutter -name *.podspec -type f -exec sed -i.bak 's/1\.10\.33/1\.10\.34/g' {} \;
find nodejs-addon-examples -name package.json -type f -exec sed -i.bak 's/1\.10\.33/1\.10\.34/g' {} \;

find harmony-os -name "README.md" -type f -exec sed -i.bak 's/1\.10\.33/1\.10\.34/g' {} \;
find harmony-os -name oh-package.json5 -type f -exec sed -i.bak 's/1\.10\.33/1\.10\.34/g' {} \;
