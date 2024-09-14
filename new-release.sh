#!/usr/bin/env bash

find flutter -name *.yaml -type f -exec sed -i.bak 's/1\.10\.25/1\.10\.26/g' {} \;
find dart-api-examples -name *.yaml -type f -exec sed -i.bak 's/1\.10\.25/1\.10\.26/g' {} \;
find flutter-examples -name *.yaml -type f -exec sed -i.bak 's/1\.10\.25/1\.10\.26/g' {} \;
find flutter -name *.podspec -type f -exec sed -i.bak 's/1\.10\.25/1\.10\.26/g' {} \;
find nodejs-addon-examples -name package.json -type f -exec sed -i.bak 's/1\.10\.25/1\.10\.26/g' {} \;
