#!/usr/bin/env bash
# Copyright (c)  2023  Xiaomi Corporation

set -ex

mkdir -p macos linux windows all

cp ./online.cs all
cp ./offline.cs all

./generate.py

pushd linux
dotnet build -c Release
dotnet pack -c Release -o ../packages
popd

pushd macos
dotnet build -c Release
dotnet pack -c Release -o ../packages
popd

pushd windows
dotnet build -c Release
dotnet pack -c Release -o ../packages
popd

pushd all
dotnet build -c Release
dotnet pack -c Release -o ../packages
popd

ls -lh packages
