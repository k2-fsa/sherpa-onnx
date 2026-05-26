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

GO_PROXY_WAIT_SECS=30
GO_PROXY_MAX_RETRIES=40

# Proactively tell the Go module proxy to fetch a specific version.
# Requesting the .info endpoint forces proxy.golang.org to fetch and cache
# the module rather than waiting for its periodic indexing crawl.
kick_go_proxy() {
  local pkg="$1"
  local version="$2"
  echo "Kicking Go proxy to fetch $pkg@$version ..."
  curl -sS "https://proxy.golang.org/${pkg}/@v/${version}.info" || true
  echo ""
}

# Wait for Go proxy to index newly published packages.
# Uses the .info endpoint which is a direct, reliable check.
wait_for_go_proxy() {
  local pkg="$1"
  local version="$2"
  local i

  kick_go_proxy "$pkg" "$version"

  for i in $(seq 1 $GO_PROXY_MAX_RETRIES); do
    echo "Attempt $i/$GO_PROXY_MAX_RETRIES: checking $pkg@$version ..."
    if curl -sS -o /dev/null -w "%{http_code}" "https://proxy.golang.org/${pkg}/@v/${version}.info" | grep -q "200"; then
      echo "  -> $pkg@$version is available on Go proxy"
      return 0
    fi
    echo "  -> not ready yet, sleeping ${GO_PROXY_WAIT_SECS}s ..."
    sleep $GO_PROXY_WAIT_SECS
  done
  echo "ERROR: $pkg@$version not available after $GO_PROXY_MAX_RETRIES attempts"
  return 1
}

# Run go mod tidy with retries. Sometimes the proxy has the module metadata
# but the zip download is still being processed.
run_go_mod_tidy() {
  local i
  for i in $(seq 1 $GO_PROXY_MAX_RETRIES); do
    echo "Attempt $i/$GO_PROXY_MAX_RETRIES: running go mod tidy ..."
    if go mod tidy 2>&1; then
      echo "  -> go mod tidy succeeded"
      return 0
    fi
    echo "  -> go mod tidy failed, sleeping ${GO_PROXY_WAIT_SECS}s ..."
    sleep $GO_PROXY_WAIT_SECS
  done
  echo "ERROR: go mod tidy failed after $GO_PROXY_MAX_RETRIES attempts"
  return 1
}

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
  wget -q https://huggingface.co/csukuangfj2/sherpa-onnx-wheels/resolve/main/cpu/$SHERPA_ONNX_VERSION/sherpa_onnx_core-${SHERPA_ONNX_VERSION}-py3-none-manylinux2014_x86_64.whl
  unzip sherpa_onnx_core-${SHERPA_ONNX_VERSION}-py3-none-manylinux2014_x86_64.whl

  rm -fv $dst/_sherpa*.so
  cp -v sherpa_onnx/lib/lib*.so* $dst

  cd ..
  rm -rf t

  rm -rf sherpa-onnx-go-linux/lib/aarch64-unknown-linux-gnu/lib*
  dst=$(realpath sherpa-onnx-go-linux/lib/aarch64-unknown-linux-gnu)
  mkdir t
  cd t
  wget -q https://huggingface.co/csukuangfj2/sherpa-onnx-wheels/resolve/main/cpu/$SHERPA_ONNX_VERSION/sherpa_onnx_core-${SHERPA_ONNX_VERSION}-py3-none-manylinux2014_aarch64.whl
  unzip ./sherpa_onnx_core-${SHERPA_ONNX_VERSION}-py3-none-manylinux2014_aarch64.whl

  rm -fv $dst/_sherpa*.so
  cp -v sherpa_onnx/lib/lib*.so* $dst

  cd ..
  rm -rf t

  rm -rf sherpa-onnx-go-linux/lib/arm-unknown-linux-gnueabihf/lib*
  dst=$(realpath sherpa-onnx-go-linux/lib/arm-unknown-linux-gnueabihf)
  mkdir t
  cd t
  wget -q https://huggingface.co/csukuangfj2/sherpa-onnx-wheels/resolve/main/cpu/$SHERPA_ONNX_VERSION/sherpa_onnx_core-$SHERPA_ONNX_VERSION-py3-none-manylinux_2_35_armv7l.whl
  unzip ./sherpa_onnx_core-$SHERPA_ONNX_VERSION-py3-none-manylinux_2_35_armv7l.whl

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
  kick_go_proxy "github.com/k2-fsa/sherpa-onnx-go-linux" "v$SHERPA_ONNX_VERSION"
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
  wget -q https://huggingface.co/csukuangfj2/sherpa-onnx-wheels/resolve/main/cpu/$SHERPA_ONNX_VERSION/sherpa_onnx_core-${SHERPA_ONNX_VERSION}-py3-none-macosx_10_15_x86_64.whl
  unzip ./sherpa_onnx_core-${SHERPA_ONNX_VERSION}-py3-none-macosx_10_15_x86_64.whl

  cp -v sherpa_onnx/lib/*.dylib $dst/

  pushd $dst
  cp -v libonnxruntime.*.dylib libonnxruntime.dylib
  popd

  cd ..
  rm -rf t

  echo "process macos arm64"
  rm -rf sherpa-onnx-go-macos/lib/aarch64-apple-darwin/lib*
  dst=$(realpath sherpa-onnx-go-macos/lib/aarch64-apple-darwin)

  mkdir t
  cd t
  wget -q https://huggingface.co/csukuangfj2/sherpa-onnx-wheels/resolve/main/cpu/$SHERPA_ONNX_VERSION/sherpa_onnx_core-${SHERPA_ONNX_VERSION}-py3-none-macosx_11_0_arm64.whl
  unzip ./sherpa_onnx_core-${SHERPA_ONNX_VERSION}-py3-none-macosx_11_0_arm64.whl

  cp -v sherpa_onnx/lib/*.dylib $dst/

  pushd $dst
  cp -v libonnxruntime.*.dylib libonnxruntime.dylib
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
  kick_go_proxy "github.com/k2-fsa/sherpa-onnx-go-macos" "v$SHERPA_ONNX_VERSION"
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
  wget -q https://huggingface.co/csukuangfj2/sherpa-onnx-wheels/resolve/main/cpu/$SHERPA_ONNX_VERSION/sherpa_onnx_core-${SHERPA_ONNX_VERSION}-py3-none-win_amd64.whl
  unzip ./sherpa_onnx_core-${SHERPA_ONNX_VERSION}-py3-none-win_amd64.whl

  cp -v sherpa_onnx/lib/*.dll $dst

  cd ..
  rm -rf t

  rm -fv sherpa-onnx-go-windows/lib/i686-pc-windows-gnu/*
  dst=$(realpath sherpa-onnx-go-windows/lib/i686-pc-windows-gnu)
  mkdir t
  cd t
  wget -q https://huggingface.co/csukuangfj2/sherpa-onnx-wheels/resolve/main/cpu/$SHERPA_ONNX_VERSION/sherpa_onnx_core-${SHERPA_ONNX_VERSION}-py3-none-win32.whl
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
  kick_go_proxy "github.com/k2-fsa/sherpa-onnx-go-windows" "v$SHERPA_ONNX_VERSION"
  rm -rf sherpa-onnx-go-windows
}

function basic() {
  echo "Process sherpa-onnx-go"
  git clone git@github.com:k2-fsa/sherpa-onnx-go.git

  python3 ./generate.py -s ./sherpa_onnx.go -o ./sherpa-onnx-go

  cd sherpa-onnx-go

  # Update go.mod to reference the new platform package versions.
  # The platform packages (linux/macos/windows) have already been published
  # and tagged with v$SHERPA_ONNX_VERSION.
  local ver="v$SHERPA_ONNX_VERSION"
  sed -i.bak \
    -e "s|github.com/k2-fsa/sherpa-onnx-go-linux .*|github.com/k2-fsa/sherpa-onnx-go-linux $ver|" \
    -e "s|github.com/k2-fsa/sherpa-onnx-go-macos .*|github.com/k2-fsa/sherpa-onnx-go-macos $ver|" \
    -e "s|github.com/k2-fsa/sherpa-onnx-go-windows .*|github.com/k2-fsa/sherpa-onnx-go-windows $ver|" \
    go.mod
  rm -f go.mod.bak

  echo "--- Updated go.mod ---"
  cat go.mod
  echo "--- end go.mod ---"

  # Wait for the Go module proxy to index all three platform packages,
  # then regenerate go.sum. The proxy (proxy.golang.org) may take
  # several minutes after a git tag push before the module is downloadable.
  local pkg
  for pkg in sherpa-onnx-go-linux sherpa-onnx-go-macos sherpa-onnx-go-windows; do
    wait_for_go_proxy "github.com/k2-fsa/$pkg" "$ver"
  done

  # go remove stale go.sum entries, then re-resolve with the new versions
  rm -f go.sum
  run_go_mod_tidy

  echo "--- Updated go.sum ---"
  cat go.sum
  echo "--- end go.sum ---"

  cd ..

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

# Publishing order matters:
#   1. Platform packages first (linux, windows, osx) — they have no inter-dependencies
#   2. Wait for Go proxy to index them
#   3. sherpa-onnx-go last — it depends on all three platform packages
linux
windows
osx
basic

rm -fv ~/.ssh/github
