#!/usr/bin/env bash

set -euo pipefail

os_name="${RUNNER_OS:-}"
if [[ -z "$os_name" ]]; then
  case "$(uname -s)" in
    Linux*) os_name="Linux" ;;
    Darwin*) os_name="macOS" ;;
    MINGW*|MSYS*|CYGWIN*) os_name="Windows" ;;
    *) os_name="$(uname -s)" ;;
  esac
fi

show_one() {
  local bin="$1"

  echo "=== Binary info: $bin ==="
  ls -lh "$bin"
  echo
  echo "=== Binary dependencies ($os_name) ==="

  case "$os_name" in
    Linux)
      ldd "$bin"
      ;;
    macOS)
      otool -L "$bin"
      ;;
    Windows)
      if command -v dumpbin >/dev/null 2>&1; then
        dumpbin /dependents "$(cygpath -w "$PWD/$bin")"
      else
        echo "dumpbin is not available on PATH"
        return 1
      fi
      ;;
    *)
      echo "Unsupported OS for dependency inspection: $os_name"
      return 1
      ;;
  esac
}

if [[ "${1:-}" == "--all" ]]; then
  shopt -s nullglob
  bins=()

  case "$os_name" in
    Windows)
      for bin in target/debug/examples/*.exe; do
        bins+=("$bin")
      done
      ;;
    *)
      for bin in target/debug/examples/*; do
        if [[ -f "$bin" && -x "$bin" && "$bin" != *.d ]]; then
          bins+=("$bin")
        fi
      done
      ;;
  esac

  if [[ ${#bins[@]} -eq 0 ]]; then
    echo "No built example executables found in target/debug/examples"
    exit 0
  fi

  printf '%s\n' "${bins[@]}" | sort | while IFS= read -r bin; do
    show_one "$bin"
    echo
  done
else
  name="${1:?Please provide an example binary name or use --all}"
  exe_suffix=""
  case "$os_name" in
    Windows) exe_suffix=".exe" ;;
  esac

  bin="target/debug/examples/${name}${exe_suffix}"

  if [[ ! -f "$bin" ]]; then
    echo "Binary not found: $bin"
    exit 1
  fi

  show_one "$bin"
fi
