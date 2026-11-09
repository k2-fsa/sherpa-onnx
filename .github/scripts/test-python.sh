#!/usr/bin/env bash
#
# Runs every Python test group sequentially.
#
# CI runs these groups in parallel as separate jobs (see
# .github/workflows/run-python-test.yaml). Use this script to run the full
# suite in one go, e.g. to reproduce a CI failure locally.

set -ex

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/common.sh"

for group in misc asr-offline asr-streaming tts; do
  log "================================================================"
  log "test group: $group"
  log "================================================================"
  $SCRIPT_DIR/test-python-$group.sh
done

log "All Python test groups passed"
