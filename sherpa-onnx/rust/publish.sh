#!/usr/bin/env bash

pushd sherpa-onnx-sys

cp -v ../../../README.md ./
cp -v ../../../LICENSE ./

popd

pushd sherpa-onnx

cp -v ../../../README.md ./
cp -v ../../../LICENSE ./

popd
