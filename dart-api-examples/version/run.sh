#!/usr/bin/env bash

set -ex

dart pub get

dart run ./bin/main.dart
