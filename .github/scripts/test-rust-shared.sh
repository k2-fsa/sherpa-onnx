#!/usr/bin/env bash

set -ex
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/common.sh"
cargo() {
  if [[ $# -gt 0 && "$1" == "run" ]]; then
    command cargo run --no-default-features --features shared "${@:2}"
  else
    command cargo "$@"
  fi
}

export -f cargo

bash ./.github/scripts/test-rust.sh
