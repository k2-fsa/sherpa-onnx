#!/usr/bin/env bash

sherpa_onnx_dir=$PWD
echo "sherpa_onnx_dir: $sherpa_onnx_dir"

src_dir=$sherpa_onnx_dir/.github/scripts/node-addon

platform=$(node -p "require('os').platform()")
arch=$(node -p "require('os').arch()")

SHERPA_ONNX_VERSION=$(grep "SHERPA_ONNX_VERSION" ./CMakeLists.txt  | cut -d " " -f 2  | cut -d '"' -f 2)
echo "SHERPA_ONNX_VERSION $SHERPA_ONNX_VERSION"

if [ -z $owner ]; then
  owner=k2-fsa
fi

sed -i.bak s/SHERPA_ONNX_VERSION/$SHERPA_ONNX_VERSION/g $src_dir/package-optional.json
sed -i.bak s/k2-fsa/$owner/g $src_dir/package-optional.json
sed -i.bak s/PLATFORM/$platform/g $src_dir/package-optional.json
sed -i.bak s/ARCH/$arch/g $src_dir/package-optional.json

git diff $src_dir/package-optional.json

dst=$sherpa_onnx_dir/sherpa-onnx-node
mkdir -p $dst

cp $src_dir/package-optional.json $dst/package.json
cp $src_dir/index.js $dst/
cp $src_dir/README-optional.md $dst/README.md

cp -fv build/install/lib/lib* $dst/ || true
cp -fv build/install/lib/*dll $dst/ || true

cp scripts/node-addon-api/build/Release/sherpa-onnx.node $dst/

ls $dst
