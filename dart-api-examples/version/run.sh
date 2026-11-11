#!/usr/bin/env bash

set -ex

dart pub get

echo "=== Sync init ==="
dart run ./bin/main_sync.dart

echo ""
echo "=== Async init ==="
dart run ./bin/main_async.dart

echo ""
echo "=== Isolate with sync init ==="
dart run ./bin/main_isolate_sync.dart

echo ""
echo "=== Isolate with async init ==="
dart run ./bin/main_isolate_async.dart
