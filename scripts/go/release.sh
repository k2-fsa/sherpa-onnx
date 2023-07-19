#!/usr/bin/env bash

set -ex

SHERPA_ONNX_VERSION=v$(grep "SHERPA_ONNX_VERSION" ./CMakeLists.txt  | cut -d " " -f 2  | cut -d '"' -f 2)

git clone git@github.com:k2-fsa/sherpa-onnx-go-linux.git

echo "Copy libs for Linux x86_64"

rm -rf sherpa-onnx-go-linux/lib/x86_64-unknown-linux-gnu/lib*
cp -v ./linux/sherpa_onnx/lib/libkaldi-native-fbank-core.so sherpa-onnx-go-linux/lib/x86_64-unknown-linux-gnu/
cp -v ./linux/sherpa_onnx/lib/libonnxruntime* sherpa-onnx-go-linux/lib/x86_64-unknown-linux-gnu/
cp -v ./linux/sherpa_onnx/lib/libsherpa-onnx-c-api.so sherpa-onnx-go-linux/lib/x86_64-unknown-linux-gnu/
cp -v ./linux/sherpa_onnx/lib/libsherpa-onnx-core.so sherpa-onnx-go-linux/lib/x86_64-unknown-linux-gnu/

echo "Copy sources for Linux x86_64"
cp sherpa-onnx/c-api/c-api.h sherpa-onnx-go-linux/sherpa-onnx/
cp scripts/go/sherpa_onnx.go sherpa-onnx-go-linux/sherpa-onnx/

cd sherpa-onnx-go-linux
tag=$(git describe --abbrev=0 --tags)
if [[ x"$VERSION" == x"auto" ]]; then
  # this is a pre-release
  if [[ $tag == ${SHERPA_ONNX_VERSION}* ]]; then
    # echo we have already release pre-release before, so just increment it
    last=$(echo $tag | rev | cut -d'.' -f 1 | rev)
    new_last=$((last+1))
    new_tag=${SHERPA_ONNX_VERSION}-alpha.${new_last}
  else
    new_tag=${SHERPA_ONNX_VERSION}-alpha.1
  fi
else
  new_tag=$VERSION
fi
echo "new_tag: $new_tag"
# git add .
# git commit -m "Rlease $new_tag"
# git push
# git tag $new_tag
# git push
#
rm -fv ~/.ssh/github
