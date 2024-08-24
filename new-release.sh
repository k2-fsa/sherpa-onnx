#!/usr/bin/env bash

find flutter -name *.yaml -type f -exec sed -i.bak 's/1\.10\.22/1\.10\.23/g' {} \;
find dart-api-examples -name *.yaml -type f -exec sed -i.bak 's/1\.10\.22/1\.10\.23/g' {} \;
find flutter-examples -name *.yaml -type f -exec sed -i.bak 's/1\.10\.22/1\.10\.23/g' {} \;
find flutter -name *.podspec -type f -exec sed -i.bak 's/1\.10\.22/1\.10\.23/g' {} \;
find nodejs-addon-examples -name package.json -type f -exec sed -i.bak 's/1\.10\.22/1\.10\.23/g' {} \;
