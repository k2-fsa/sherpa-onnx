#!/usr/bin/env bash

set -ex

cargo() {
  if [[ $# -gt 0 && "$1" == "run" ]]; then
    command cargo run --features static "${@:2}"
  else
    command cargo "$@"
  fi
}

export -f cargo

./.github/scripts/test-rust.sh
