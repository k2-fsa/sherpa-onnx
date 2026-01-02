#!/usr/bin/env bash

set -ex

git config --global user.email "csukuangfj@gmail.com"
git config --global user.name "Fangjun Kuang"

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
SHERPA_ONNX_DIR=$(realpath $SCRIPT_DIR/../..)
echo "SCRIPT_DIR: $SCRIPT_DIR"
echo "SHERPA_ONNX_DIR: $SHERPA_ONNX_DIR"


SHERPA_ONNX_VERSION=$(grep "SHERPA_ONNX_VERSION" $SHERPA_ONNX_DIR/CMakeLists.txt  | cut -d " " -f 2  | cut -d '"' -f 2)
echo "SHERPA_ONNX_VERSION $SHERPA_ONNX_VERSION"

function linux() {
  echo "Process linux"
  git clone git@github.com:k2-fsa/sherpa-onnx-go-linux.git

  rm -v ./sherpa-onnx-go-linux/*.go

  cp -v ./sherpa_onnx.go ./sherpa-onnx-go-linux/
  cp -v ./_internal/c-api.h ./sherpa-onnx-go-linux/
  cp -v ./_internal/build_linux_*.go ./sherpa-onnx-go-linux/

  rm -rf sherpa-onnx-go-linux/lib/x86_64-unknown-linux-gnu/lib*
  dst=$(realpath sherpa-onnx-go-linux/lib/x86_64-unknown-linux-gnu)
  mkdir t
  cd t
  wget -q https://huggingface.co/csukuangfj/sherpa-onnx-wheels/resolve/main/cpu/$SHERPA_ONNX_VERSION/sherpa_onnx_core-${SHERPA_ONNX_VERSION}-py3-none-manylinux2014_x86_64.whl
  unzip sherpa_onnx_core-${SHERPA_ONNX_VERSION}-py3-none-manylinux2014_x86_64.whl

  rm -fv $dst/_sherpa*.so
  cp -v sherpa_onnx/lib/lib*.so* $dst

  cd ..
  rm -rf t

  rm -rf sherpa-onnx-go-linux/lib/aarch64-unknown-linux-gnu/lib*
  dst=$(realpath sherpa-onnx-go-linux/lib/aarch64-unknown-linux-gnu)
  mkdir t
  cd t
  wget -q https://huggingface.co/csukuangfj/sherpa-onnx-wheels/resolve/main/cpu/$SHERPA_ONNX_VERSION/sherpa_onnx_core-${SHERPA_ONNX_VERSION}-py3-none-manylinux2014_aarch64.whl
  unzip ./sherpa_onnx_core-${SHERPA_ONNX_VERSION}-py3-none-manylinux2014_aarch64.whl

  rm -fv $dst/_sherpa*.so
  cp -v sherpa_onnx/lib/lib*.so* $dst

  cd ..
  rm -rf t

  rm -rf sherpa-onnx-go-linux/lib/arm-unknown-linux-gnueabihf/lib*
  dst=$(realpath sherpa-onnx-go-linux/lib/arm-unknown-linux-gnueabihf)
  mkdir t
  cd t
  wget -q https://huggingface.co/csukuangfj/sherpa-onnx-wheels/resolve/main/cpu/$SHERPA_ONNX_VERSION/sherpa_onnx-${SHERPA_ONNX_VERSION}-cp38-cp38-linux_armv7l.whl
  unzip ./sherpa_onnx-${SHERPA_ONNX_VERSION}-cp38-cp38-linux_armv7l.whl

  rm -fv $dst/_sherpa*.so
  cp -v sherpa_onnx/lib/lib*.so* $dst

  cd ..
  rm -rf t

  echo "------------------------------"
  cd sherpa-onnx-go-linux
  git status
  git add .
  git commit -m "Release v$SHERPA_ONNX_VERSION" && \
  git push && \
  git tag v$SHERPA_ONNX_VERSION && \
  git push origin v$SHERPA_ONNX_VERSION || true
  cd ..
  rm -rf sherpa-onnx-go-linux
}

function osx() {
  echo "Process osx-x64"
  git clone git@github.com:k2-fsa/sherpa-onnx-go-macos.git
  rm -v ./sherpa-onnx-go-macos/*.go
  cp -v ./sherpa_onnx.go ./sherpa-onnx-go-macos/
  cp -v ./_internal/c-api.h ./sherpa-onnx-go-macos/
  cp -v ./_internal/build_darwin_*.go ./sherpa-onnx-go-macos/

  rm -rf sherpa-onnx-go-macos/lib/x86_64-apple-darwin/lib*
  dst=$(realpath sherpa-onnx-go-macos/lib/x86_64-apple-darwin/)

  mkdir t
  cd t
  wget -q https://huggingface.co/csukuangfj/sherpa-onnx-wheels/resolve/main/cpu/$SHERPA_ONNX_VERSION/sherpa_onnx_core-${SHERPA_ONNX_VERSION}-py3-none-macosx_10_15_x86_64.whl
  unzip ./sherpa_onnx_core-${SHERPA_ONNX_VERSION}-py3-none-macosx_10_15_x86_64.whl

  cp -v sherpa_onnx/lib/*.dylib $dst/

  pushd $dst
  cp -v libonnxruntime.1.17.1.dylib libonnxruntime.dylib
  popd

  cd ..
  rm -rf t

  echo "process macos arm64"
  rm -rf sherpa-onnx-go-macos/lib/aarch64-apple-darwin/lib*
  dst=$(realpath sherpa-onnx-go-macos/lib/aarch64-apple-darwin)

  mkdir t
  cd t
  wget -q https://huggingface.co/csukuangfj/sherpa-onnx-wheels/resolve/main/cpu/$SHERPA_ONNX_VERSION/sherpa_onnx_core-${SHERPA_ONNX_VERSION}-py3-none-macosx_11_0_arm64.whl
  unzip ./sherpa_onnx_core-${SHERPA_ONNX_VERSION}-py3-none-macosx_11_0_arm64.whl

  cp -v sherpa_onnx/lib/*.dylib $dst/

  pushd $dst
  cp -v libonnxruntime.1.17.1.dylib libonnxruntime.dylib
  popd

  cd ..
  rm -rf t
  echo "------------------------------"
  cd sherpa-onnx-go-macos
  git status
  git add .
  git commit -m "Release v$SHERPA_ONNX_VERSION" && \
  git push && \
  git tag v$SHERPA_ONNX_VERSION && \
  git push origin v$SHERPA_ONNX_VERSION || true
  cd ..
  rm -rf sherpa-onnx-go-macos
}

function windows() {
  echo "Process windows"
  git clone git@github.com:k2-fsa/sherpa-onnx-go-windows.git
  rm -v ./sherpa-onnx-go-windows/*.go
  cp -v ./sherpa_onnx.go ./sherpa-onnx-go-windows/
  cp -v ./_internal/c-api.h ./sherpa-onnx-go-windows/
  cp -v ./_internal/build_windows_*.go ./sherpa-onnx-go-windows/

  rm -fv sherpa-onnx-go-windows/lib/x86_64-pc-windows-gnu/*
  dst=$(realpath sherpa-onnx-go-windows/lib/x86_64-pc-windows-gnu)
  mkdir t
  cd t
  wget -q https://huggingface.co/csukuangfj/sherpa-onnx-wheels/resolve/main/cpu/$SHERPA_ONNX_VERSION/sherpa_onnx_core-${SHERPA_ONNX_VERSION}-py3-none-win_amd64.whl
  unzip ./sherpa_onnx_core-${SHERPA_ONNX_VERSION}-py3-none-win_amd64.whl

  cp -v sherpa_onnx/lib/*.dll $dst

  cd ..
  rm -rf t

  rm -fv sherpa-onnx-go-windows/lib/i686-pc-windows-gnu/*
  dst=$(realpath sherpa-onnx-go-windows/lib/i686-pc-windows-gnu)
  mkdir t
  cd t
  wget -q https://huggingface.co/csukuangfj/sherpa-onnx-wheels/resolve/main/cpu/$SHERPA_ONNX_VERSION/sherpa_onnx_core-${SHERPA_ONNX_VERSION}-py3-none-win32.whl
  unzip ./sherpa_onnx_core-${SHERPA_ONNX_VERSION}-py3-none-win32.whl

  cp -v sherpa_onnx/lib/*.dll $dst

  cd ..
  rm -rf t
  echo "------------------------------"
  cd sherpa-onnx-go-windows
  git status
  git add .
  git commit -m "Release v$SHERPA_ONNX_VERSION" && \
  git push && \
  git tag v$SHERPA_ONNX_VERSION && \
  git push origin v$SHERPA_ONNX_VERSION || true
  cd ..
  rm -rf sherpa-onnx-go-windows
}

# parse golang file and generate struct defines into platform file
# params：
# $1 = source file（like: ./sherpa.go）
# $2 = output folder（like: ./model）
# $3 = platform name（like: windows、linux、darwin）
update() {
    local source_file="$1"
    local output_dir="$2"
    local platform="$3"

    if [ -z "$source_file" ] || [ -z "$output_dir" ] || [ -z "$platform" ]; then
        echo "error:param is invalid" >&2
        echo "command pattern: update source_file output_dir platform" >&2
        echo "example：update ./sherpa.go . \"windows\"" >&2
        return 1
    fi
    if [[ "$source_file" != *.go ]]; then
        echo "error：source file $source_file is not golang file！" >&2
        return 1
    fi
    if [ ! -f "$source_file" ]; then
        echo "error：source file $source_file is not exist！" >&2
        return 1
    fi

    local source_filename=$(basename "$source_file")
    local target_filename="sherpa_onnx_${platform}.go"
    local target_file="${output_dir}/sherpa_onnx/${target_filename}"
    local target_dir=$(dirname "$target_file")

    mkdir -p "$target_dir" >/dev/null 2>&1
    if [ ! -d "$target_dir" ]; then
        echo "error：create folder $target_dir failed！" >&2
        return 1
    fi

    case $platform in
      "windows")
        platform_content="//go:build (windows && amd64) || (windows && 386)"
        ;;
      "linux")
        platform_content="//go:build (!android && linux && arm64) || (!android && linux && amd64 && !musl) || (!android && linux && arm && !arm7) || (!android && arm7) || (!android && linux && 386 && !musl) || (!android && musl) || (!android && linux && mips) || (!android && linux && mips64) || (!android && linux && mips64le) || (!android && linux && mipsle)"
        ;;
      "macos")
        platform_content="//go:build (darwin && amd64 && !ios) || (darwin && arm64 && !ios)"
        ;;
      *)
        platform_content=""
        ;;
    esac

    cat > "$target_file" << EOF
$platform_content
package sherpa_onnx

// ============================================================
// Code Generated Automatically for $platform platform, DO NOT EDIT MANUALLY!!
// ============================================================

import (
	sherpa "github.com/k2-fsa/sherpa-onnx-go-$platform"
)

// ============================================================
// Structs
// ============================================================

EOF
    grep -nE 'type\s+[A-Z][A-Za-z0-9_]*\s+struct\s*\{|type\s+[A-Z][A-Za-z0-9_]*\s+=\s+struct\s*{|type\s+[A-Z][A-Za-z0-9_]*\s+=\s*' "$source_file" \
        | sed -E 's/^[0-9]+://; s/^\s*//; s/\s+/ /g; s/type ([^ ]+) .*struct.*/\1/; s/type ([^ =]+) .*=.*/\1/' \
        | sort -u \
        | awk '{print "type " $0 " = sherpa." $0}' \
        >> "$target_file"

    cat >> "$target_file" << EOF

// ============================================================
// Functions
// ============================================================

EOF
    grep -nE 'func\s+[A-Z][^ (]*\s*\(' "$source_file" \
        | sed -E 's/^[0-9]+://; s/^\s*//; s/\s+/ /g; s/func ([^ (]+).*/\1/' \
        | sort -u \
        | awk '{print "var " $0 " = sherpa." $0}' \
        >> "$target_file"

    echo "succeed！"
    echo "source：$source_file"
    echo "platform：$platform"
    echo "target：$target_file"
    return 0
}

function basic() {
  echo "Process basic"
  git clone git@github.com:k2-fsa/sherpa-onnx-go.git

#  update ./sherpa_onnx.go ./sherpa-onnx-go "windows"
#  update ./sherpa_onnx.go ./sherpa-onnx-go "linux"
#  update ./sherpa_onnx.go ./sherpa-onnx-go "macos"

  python3 ./generate.py -s ./sherpa_onnx.go -o ./sherpa-onnx-go

  echo "------------------------------"
  cd sherpa-onnx-go
  git status
  git add .
  git commit -m "Release v$SHERPA_ONNX_VERSION" && \
    git push && \
    git tag v$SHERPA_ONNX_VERSION && \
    git push origin v$SHERPA_ONNX_VERSION
  cd ..
  rm -rf sherpa-onnx-go
}

basic
windows
linux
osx

rm -fv ~/.ssh/github
