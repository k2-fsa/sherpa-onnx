#!/usr/bin/env bash
set -ex
cd go-api-examples/non-streaming-tts-phonemize
go mod tidy

# On Windows, piper-phonemize's native DLLs live inside the
# piper-phonemize-go-windows module (the module cache) and are not on the
# loader's search path. Copy them next to the built binary so they can be
# found at runtime. (Linux/macOS find the lib via the normal mechanisms.)
if [[ "$OSTYPE" == msys* || "$OSTYPE" == cygwin* || "$OSTYPE" == win32 ]]; then
  case "$(go env GOARCH)" in
    amd64) triple=x86_64-pc-windows-gnu ;;
    386)   triple=i686-pc-windows-gnu ;;
    arm64) triple=aarch64-pc-windows-gnu ;;
  esac
  pp_dir=$(go list -m -f '{{.Dir}}' github.com/csukuangfj/piper-phonemize-go-windows)
  cp -v "$pp_dir/lib/$triple"/*.dll .
fi

go build
./run-vits-piper-phonemize.sh
./run-kitten-en-phonemize.sh
./run-kokoro-en-phonemize.sh
./run-matcha-en-phonemize.sh
